import os
import warnings
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDIMScheduler
import numpy as np
from tqdm import tqdm
import argparse
import logging

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="SimWorld Inference Script")
    parser.add_argument("--sd_path", type=str, required=True, help="Path to Stable Diffusion model")
    parser.add_argument("--cn_path", type=str, required=True, help="Path to ControlNet model")
    parser.add_argument("--controlnet_ckpt", type=str, required=True, help="Path to ControlNet checkpoint weights")
    parser.add_argument("--source_image", type=str, required=True, help="Path to conditional image (e.g., segmentation map)")
    parser.add_argument("--target_prompt", type=str, required=True, help="Path to text prompt file")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA device ID to use")
    parser.add_argument("--time_steps", type=int, default=100, help="Number of inference timesteps")
    parser.add_argument("--output_image", type=str, default="output.png", help="Path to save the generated image")
    return parser.parse_args()

def load_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def preprocess_image(image_path, size=(512, 512), normalize=True):
    transform_list = [transforms.Resize(size), transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def decode_latents(latents, vae, device):
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading tokenizer and encoding prompt...")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    prompt_text = load_prompt(args.target_prompt)
    input_ids = tokenizer(prompt_text, max_length=tokenizer.model_max_length, padding="max_length",
                          truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(device)
        encoder_hidden_states = text_encoder(input_ids)[0]

    logging.info("Loading conditional image...")
    source_image = preprocess_image(args.source_image, normalize=False).to(device)

    logging.info("Loading models...")
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_pretrained(args.cn_path).to(device)
    controlnet_ckpt = torch.load(args.controlnet_ckpt, map_location=device)
    controlnet.load_state_dict(controlnet_ckpt, strict=False)

    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    scheduler.set_timesteps(args.time_steps)

    height, width = 512, 512
    latents = torch.randn((1, unet.in_channels, height // 8, width // 8)).to(device)

    logging.info("Starting inference...")
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps):
            down_res_samples, mid_res_sample = controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=source_image,
                return_dict=False,
            )

            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_res_samples,
                mid_block_additional_residual=mid_res_sample,
                return_dict=False,
            )[0]

            latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

    del unet, controlnet
    torch.cuda.empty_cache()

    logging.info("Loading VAE decoder and decoding latents...")
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(device)
    images = decode_latents(latents, vae, device)
    images = images[0].resize((960,600),resample=Image.LANCZOS)
    logging.info(f"Saving generated image to {args.output_image}")
    images.save(args.output_image)
    logging.info("Inference completed successfully.")

if __name__ == "__main__":
    main()
