# OFT (Orthogonal Fine-Tuning)

Diffusers Implementation of Controlling Text-to-Image Diffusion by Orthogonal Finetuning ([https://arxiv.org/pdf/2306.07280.pdf](https://arxiv.org/pdf/2306.07280.pdf))

## Setup (tested on Linux with an Nvidia GPU)

1. Install PyTorch
2. `pip install -r requirements.txt`

## Usage

### Training

1. Create folders `config` and `output` if they don't already exist.
2. Create a config file at `config/config.json`. See `example.config.json` for an example. Right now it only supports fine-tuning Diffusers models.
3. Run `python train.py`

### Merging

To merge the adapter weight with a base model, run `python merge_to_original.py --processor_path /path/to/attn_processors.pt --output_path /path/to/merged_model.ckpt --model_path /path/to/base_model.ckpt`.

Right now this file only supports original Stable Diffusion (non-Diffusers) model as base models.

## Implementation Details

- Parameterized each skew-symmetric matrix as a weight matrix minus its transpose.
- For constrained orthogonal fine-tuning (COFT), the norm of the weights of each skew-symmetric matrix is given a maximum value.
- All matrices are kept in the block-diagonal form for as many operations as possible for efficiency.
