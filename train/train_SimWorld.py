import os
import gc
import json
import torch
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from torchvision import transforms
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torch.optim.lr_scheduler import OneCycleLR
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--sd_path', type=str, required=True)
    parser.add_argument('--cn_path', type=str, required=True)
    parser.add_argument('--dataset_json', type=str, required=True)
    parser.add_argument('--pretrained_controlnet', type=str, required=False)
    parser.add_argument('--save_directory', type=str, default='./models')
    parser.add_argument('--log_directory', type=str, default='./logs')
    parser.add_argument('--cuda_devices', type=str, default="0")
    parser.add_argument('--port', type=str, default="29515")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

if not os.path.exists(args.log_directory):
    os.makedirs(args.log_directory)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return SummaryWriter(log_dir=args.log_directory) if rank == 0 else None


def cleanup():
    dist.destroy_process_group()


class MyDataset(Dataset):
    def __init__(self, dataset_json):
        self.data = []
        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        with open(dataset_json, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def convert_yolo_to_bbox(self, labels, img_width, img_height):
        return [[int((x - w / 2) * img_width), int((y - h / 2) * img_height),
                 int((x + w / 2) * img_width), int((y + h / 2) * img_height)] for x, y, w, h in labels]

    def __getitem__(self, idx):
        item = self.data[idx]
        source = self.conditioning_image_transforms(Image.open(item['source']).convert('RGB'))
        target = self.image_transforms(Image.open(item['target']).convert('RGB'))
        position = [list(map(float, label[1:])) for label in item['labels']]
        bounding_boxes = self.convert_yolo_to_bbox(position, target.shape[1], target.shape[2])
        return {'pixel_values': target, 'prompt': item['prompts'],
                'conditioning_pixel_values': source, 'bounding_boxes': bounding_boxes}


def calculate_loss_weights(image_size, bounding_boxes, device, min_weight, max_weight, global_step, total_steps, pct_start=0.3):
    batch_size = len(bounding_boxes)
    weights = torch.ones((batch_size, 1, *image_size), device=device)
    progress = global_step / total_steps
    cosine_progress = (progress / pct_start if progress <= pct_start else (progress - pct_start) / (1 - pct_start))
    current_weight = min_weight + (max_weight - min_weight) * (1 - math.cos(math.pi * cosine_progress)) / 2
    for idx, box_set in enumerate(bounding_boxes):
        for box in box_set:
            x1, y1, x2, y2 = map(int, box)
            weights[idx, 0, y1:y2, x1:x2] = current_weight
    return weights


def collate_fn(data):
    return {'pixel_values': torch.stack([i['pixel_values'] for i in data]),
            'conditioning_pixel_values': torch.stack([i['conditioning_pixel_values'] for i in data]),
            'bounding_boxes': [i['bounding_boxes'] for i in data], 'prompts': [i['prompt'] for i in data]}


def train(rank, world_size, num_epochs=100):
    writer = setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder='text_encoder').to(device)
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder='vae').to(device)
    unet = UNet2DConditionModel.from_pretrained(args.sd_path, subfolder='unet').to(device)
    controlnet = ControlNetModel.from_pretrained(args.cn_path, torch_dtype=torch.float32).to(device)
    if args.pretrained_controlnet:
        controlnet.load_state_dict(torch.load(args.pretrained_controlnet))
    controlnet.train()
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.requires_grad_(True)

    controlnet = DDP(controlnet, device_ids=[rank])
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=2e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    dataset = MyDataset(args.dataset_json)
    train_sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, sampler=train_sampler, num_workers=4, pin_memory=True)

    num_samples = len(train_loader.dataset)
    num_gpus = world_size
    accumulation_steps = 8
    bs = train_loader.batch_size
    effective_batch_size = bs * num_gpus * accumulation_steps
    steps_per_epoch = num_samples // effective_batch_size
    total_steps = steps_per_epoch * num_epochs

    lr_scheduler = OneCycleLR(optimizer, max_lr=2e-4, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    ema = ExponentialMovingAverage(controlnet.parameters(), decay=0.995)
    global_step = 0

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            data = {k: (v.to(device).to(torch.float32) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}

            with torch.no_grad():
                input_ids = tokenizer.batch_encode_plus(data['prompts'], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(data['pixel_values']).latent_dist.sample() * 0.18215
                controlnet_image = data['conditioning_pixel_values']
                bsz = latents.shape[0]
                timesteps = torch.randint(0, 1000, (bsz,), device=device).long()

            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[sample for sample in down_block_res_samples],
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            latent_size = model_pred.shape[-2:]
            weights = F.interpolate(
                calculate_loss_weights(data['pixel_values'].shape[2:], data['bounding_boxes'], device, 1.0, 2.0, global_step, total_steps),
                size=latent_size, mode='bilinear', align_corners=False
            )

            weights = weights.expand(-1, model_pred.shape[1], -1, -1)
            weighted_loss = (F.mse_loss(model_pred, noise, reduction='none') * weights).mean()
            weighted_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update(controlnet.parameters())
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0:
                    writer.add_scalar('Loss/train', weighted_loss.item(), global_step)
                    writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], global_step)
                    writer.flush()

                ema.copy_to(controlnet.parameters())
            epoch_loss += weighted_loss.item()
        epoch_loss /= len(train_loader.dataset)

        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}", flush=True)
            if epoch % 1 == 0 or epoch == num_epochs - 1:
                torch.save(controlnet.module.state_dict() if hasattr(controlnet, 'module') else controlnet.state_dict(), f'{args.save_directory}/epoch_{epoch+1}.model')

    cleanup()


if __name__ == '__main__':
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
