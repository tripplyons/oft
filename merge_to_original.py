
import argparse
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--processor_path', type=str)
argparser.add_argument('--output_path', type=str)
argparser.add_argument('--model_path', type=str)

args = argparser.parse_args()

processor_path = args.processor_path
output_path = args.output_path
model_path = args.model_path

processor = torch.load(processor_path, map_location='cpu')
state_dict = torch.load(model_path, map_location='cpu')

# from https://gist.github.com/jachiam/8a5c0b607e38fcc585168b90c686eb05

layer_types_to_processor = {
    'to_q': 'W_Q',
    'to_k': 'W_K',
    'to_v': 'W_V',
}

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(4):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i < 3:
            # no attention layers in down_blocks.3
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(3):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i > 0:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

def get_name_for_processor(name):
    for sd_name, hf_name in unet_conversion_map:
        name = name.replace(sd_name, hf_name)
    for sd_name, hf_name in unet_conversion_map_layer:
        name = name.replace(sd_name, hf_name)
    return name

for k in state_dict.keys():
    if ".attn2." in k and 'to_out' not in k and '.weight' in k and 'model.diffusion_model.' in k:
        # print(k, v)
        layer = '.'.join(k.split(".")[:-2]).replace('model.diffusion_model.', '')
        processor_key = get_name_for_processor(layer) + '.processor'
        weights = processor['weights'][processor_key]
        r = processor["parameters"][processor_key]["r"]
        hidden_size = processor["parameters"][processor_key]["hidden_size"]
        block_size = hidden_size // r

        layer_type = k.split(".")[-2]
        processor_name = layer_types_to_processor[layer_type]

        W = weights[processor_name]
        block_Q = W - W.transpose(1, 2)
        norm_Q = torch.norm(block_Q.flatten())
        new_norm_Q = torch.clamp(norm_Q, max=processor["parameters"][processor_key]["constraint"])
        block_Q = block_Q * ((new_norm_Q + 1e-8) / (norm_Q + 1e-8))
        I = torch.eye(block_size, device=W.device).unsqueeze(0).repeat(r, 1, 1)
        block_R = torch.matmul(I + block_Q, (I - block_Q).inverse())
        R = torch.block_diag(*block_R).to(torch.float32)
        state_dict[k] = torch.matmul(R.T, state_dict[k].to(torch.float32)).to(state_dict[k].dtype)

torch.save({
    'state_dict': state_dict
}, output_path)
