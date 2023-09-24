# based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py

import os
import json
import torch
import math
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import Dataset
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.optimization import get_scheduler
import numpy as np
import random
import pandas as pd
from PIL import Image
import torch.nn.functional as F

from attention_processor import OFTCrossAttnProcessor, save_attn_processors, load_attn_processors

# parameters

config_path = 'config'
output_dir = 'output'
gradient_accumulation_steps = 1
revision = None
mixed_precision = 'fp16'
weight_dtype = torch.float16
learning_rate = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 0
adam_epsilon = 1e-8
cache_dir = 'cache'
resolution = 512
train_batch_size = 1
dataloader_num_workers = 1
num_train_epochs = 200
max_grad_norm = 1.0
lr_warmup_steps = 100
lr_scheduler_type = 'constant'
checkpointing_steps = 100
validation_epochs = 20
num_validation_images = 4
resume_load_path = None
use_8bit_optimizer = True

with open(os.path.join(config_path, "config.json")) as f:
    config = json.load(f)

validation_prompt = config['validation_prompt']
validation_negative_prompt = config['validation_negative_prompt']
validation_cfg = config['validation_cfg']
model_name = config['model_name']

def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )

    # models

    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", revision=revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", revision=revision
    )

    # only train the attention processors

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # move to GPU

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # create or load attention processors

    if resume_load_path is not None:
        load_attn_processors(unet, 'cuda', torch.float32, resume_load_path)
    else:
        oft_attn_procs = {}
        for name in unet.attn_processors.keys():
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            oft_attn_procs[name] = OFTCrossAttnProcessor(
                hidden_size=hidden_size,
            ).to(torch.float32)

        unet.set_attn_processor(oft_attn_procs)

    if use_8bit_optimizer:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    oft_layers = AttnProcsLayers(unet.attn_processors)

    optimizer = optimizer_cls(
        oft_layers.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # load dataset and transforms

    # dataset

    def tokenize_captions(examples):
        captions = [example['caption'] for example in examples]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        dataset = {}
        image_paths = [os.path.join(config_path, example['path']) for example in examples]
        images = [Image.open(image_path) for image_path in image_paths]
        images = [image.convert("RGB") for image in images]
        dataset["pixel_values"] = torch.stack([
            train_transforms(image) for image in images
        ])
        dataset["input_ids"] = tokenize_captions(examples)
        return dataset

    with accelerator.main_process_first():
        # Set the training transforms
        examples = config['examples']
        train_dataset = preprocess_train(examples)

    tensor_dataset = torch.utils.data.TensorDataset(
        train_dataset["pixel_values"], train_dataset["input_ids"]
    )

    train_dataloader = torch.utils.data.DataLoader(
        tensor_dataset,
        shuffle=True,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    oft_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        oft_layers, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune")

    global_step = 0
    first_epoch = 0

    total_batch_size = train_batch_size * \
        accelerator.num_processes * gradient_accumulation_steps
    progress_bar = tqdm(range(global_step, max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # training loop

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, (pixel_values, input_ids) in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(pixel_values.to(
                    dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = oft_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        # also save the attention processors by themselves
                        save_attn_processors(unet, 'cuda', torch.float32,
                             os.path.join(save_path, "attn_processors.pt"))


            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        # validation check

        if accelerator.is_main_process:
            if validation_prompt is not None and (epoch + 1) % validation_epochs == 0:
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    model_name,
                    unet=accelerator.unwrap_model(unet),
                    revision=revision,
                    torch_dtype=weight_dtype,
                    safety_checker=None
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(
                    device=accelerator.device)
                images = []
                for _ in range(num_validation_images):
                    images.append(
                        pipeline(
                            validation_prompt, negative_prompt=validation_negative_prompt, guidance_scale=validation_cfg,
                            num_inference_steps=30, generator=generator).images[0]
                    )

                # save images
                save_path = os.path.join(
                    output_dir, f"validation-images-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                for i, image in enumerate(images):
                    image.save(os.path.join(save_path, f"{i}.png"))

                del pipeline
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # save attention processors
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        save_attn_processors(unet, 'cuda', torch.float32,
                             os.path.join(output_dir, "attn_processors.pt"))

    # free memory
    del unet
    del vae
    del text_encoder
    del optimizer
    torch.cuda.empty_cache()

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        model_name, revision=revision, torch_dtype=weight_dtype, safety_checker=None
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    load_attn_processors(pipeline.unet, 'cuda', torch.float32,
                         os.path.join(output_dir, "attn_processors.pt"))

    # run inference
    generator = torch.Generator(device=accelerator.device)
    images = []
    for _ in range(num_validation_images):
        images.append(pipeline(validation_prompt, negative_prompt=validation_negative_prompt, guidance_scale=validation_cfg,
                      num_inference_steps=30, generator=generator).images[0])

    # save images
    save_path = os.path.join(output_dir, f"final-images")
    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{i}.png"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
