import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers.utils import load_image
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SimWorld inference script")
    parser.add_argument("--sd_path", type=str, required=True, help="Stable Diffusion XL base model directory")
    parser.add_argument("--controlnet_path", type=str, required=True, help="ControlNet model directory")
    parser.add_argument("--vae_path", type=str, required=True, help="VAE model directory")
    parser.add_argument("--controlnet_ckpt", type=str, required=True, help="ControlNet checkpoint path")
    parser.add_argument("--input_image", type=str, required=True, help="Path to conditional input image (e.g. depth map)")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt text file")
    parser.add_argument("--cuda_device", type=str, default="0", help="Device to run inference on (e.g. cuda:0, cpu). If None, auto-detect.")
    parser.add_argument("--output_image", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of scheduler timesteps")
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizers and text encoders
    tokenizer1 = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer')
    tokenizer2 = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer_2')

    text_encoder1 = CLIPTextModel.from_pretrained(args.sd_path, subfolder='text_encoder', torch_dtype=torch.float16).to(device)
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(args.sd_path, subfolder='text_encoder_2', torch_dtype=torch.float16).to(device)

    # Load VAE, UNet, ControlNet
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder='unet', torch_dtype=torch.float16).to(device)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16).to(device)

    # Load controlnet checkpoint weights
    state_dict = torch.load(args.controlnet_ckpt, map_location=device, weights_only=True)
    controlnet.load_state_dict(state_dict)
    
    # Freeze models to avoid gradients
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder1.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    controlnet.requires_grad_(False)

    # Image preprocessing
    condition_image_transforms = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])

    # Load and preprocess conditional image
    depth_image = load_image(args.input_image)
    depth_image = condition_image_transforms(depth_image).unsqueeze(0).to(torch.float16).to(device)

    # Load prompt text
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_text = f.read().strip()

    # Prepare tokenizers and encoders list
    tokenizers = [tokenizer1, tokenizer2]
    text_encoders = [text_encoder1, text_encoder2]

    # Encode prompt with both tokenizers and text encoders
    prompt_embeds_list = []
    pooled_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        input_ids = tokenizer(prompt_text,
                              padding="max_length",
                              max_length=tokenizer.model_max_length,
                              truncation=True,
                              return_tensors="pt").input_ids.to(device)
        output = text_encoder(input_ids, output_hidden_states=True)
        pooled_embeds_list.append(output[0])  # pooled output
        prompt_embeds_list.append(output.hidden_states[-2])  # penultimate hidden layer

    # Concatenate embeddings along last dimension
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)  # shape: [batch, seq_len, combined_dim]

    # Prepare additional conditioning tensors
    add_time_ids = list((1024, 1024) + (0, 0) + (1024, 1024))
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32).to(device).repeat(depth_image.size(0), 1)

    # Prepare latents
    latents = torch.randn((1, 4, 128, 128), device=device, dtype=torch.float16)

    # Scheduler setup
    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device=device)

    # Inference loop
    for t in tqdm(scheduler.timesteps):
        down_res_samples, mid_res_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            controlnet_cond=depth_image,
            added_cond_kwargs={"text_embeds": pooled_embeds_list[1], "time_ids": add_time_ids},
            return_dict=False,
        )

        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            down_block_additional_residuals=down_res_samples,
            mid_block_additional_residual=mid_res_sample,
            added_cond_kwargs={"text_embeds": pooled_embeds_list[1], "time_ids": add_time_ids},
            return_dict=False,
        )[0]

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode latents
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
    image = image.cpu().permute(0, 2, 3, 1).detach().float().numpy()[0]

    # Convert to PIL Image and resize
    image = (image * 255).round().astype(np.uint8)
    pil_image = Image.fromarray(image).resize((960, 600), resample=Image.LANCZOS)

    # Save output
    pil_image.save(args.output_image)
    print(f"Image saved to {args.output_image}")

if __name__ == "__main__":
    main()
