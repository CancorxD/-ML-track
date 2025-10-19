import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

dataset_path = '/root/projects/lora_dataset'
image_dir = os.path.join(dataset_path, 'images')
caption_dir = os.path.join(dataset_path, 'captions')
output_dir = os.path.join(dataset_path, 'aldar_kose_2d_final')
os.makedirs(output_dir, exist_ok=True)

TOTAL_STEPS = 400
BATCH_SIZE = 1
LR = 1e-6
RANK = 16
DROPOUT = 0.1
WEIGHT_DECAY = 0.05

class CustomDataset(Dataset):
    def __init__(self, image_dir, caption_dir, target_size=(1024, 1024)):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        valid_pairs = []
        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            caption_file = os.path.join(caption_dir, f"{base_name}.txt")
            if os.path.exists(caption_file):
                valid_pairs.append((img_path, caption_file))
        self.image_paths, self.caption_paths = zip(*valid_pairs)
        print(f"Dataset: {len(self.image_paths)} images")
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB').resize(self.target_size, Image.LANCZOS)
        image_array = np.array(image).astype(np.float32) / 255.0 * 2.0 - 1.0
        latents = torch.from_numpy(image_array).permute(2, 0, 1)
        with open(self.caption_paths[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        text = f"aldar kose, beardless trickster, sly smile, big nose, green hat, orange robe, {caption}, 2d anime style, flat colors, cartoon illustration, no 3d rendering"
        return {'latents': latents, 'caption_text': text}

print("Training Aldar Köse 2D model...")
base_model_id = "cagliostrolab/animagine-xl-3.1"
pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, use_safetensors=True)
unet = pipe.unet
unet.to("cuda", dtype=torch.float16)
unet.enable_gradient_checkpointing()

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=DROPOUT,
    bias="none"
)
unet = get_peft_model(unet, lora_config)
print(f"LoRA: {unet.num_parameters()/1e6:.1f}M params")

accelerator = Accelerator(mixed_precision="fp16")
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=40, num_training_steps=TOTAL_STEPS)

dataset = CustomDataset(image_dir, caption_dir)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
unet, optimizer, data_loader, lr_scheduler = accelerator.prepare(unet, optimizer, data_loader, lr_scheduler)

add_time_ids = torch.tensor([1024, 1024, 0, 0, 1024, 1024], dtype=torch.float16).to("cuda")

global_step = 0
best_loss = float('inf')
start_time = time.time()

for epoch in range(40):
    for batch in data_loader:
        if global_step >= TOTAL_STEPS:
            break
        with accelerator.accumulate(unet):
            latents = batch["latents"].to("cuda", dtype=torch.float16)
            with torch.no_grad():
                latents = pipe.vae.encode(latents).latent_dist.sample() * pipe.vae.config.scaling_factor

            text_inputs = batch["caption_text"]
            tokenized = pipe.tokenizer(text_inputs, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            tokenized_2 = pipe.tokenizer_2(text_inputs, padding="max_length", max_length=pipe.tokenizer_2.model_max_length, truncation=True, return_tensors="pt")
            input_ids = tokenized.input_ids.to("cuda")
            input_ids_2 = tokenized_2.input_ids.to("cuda")

            with torch.no_grad():
                prompt_embeds_1 = pipe.text_encoder(input_ids).last_hidden_state
                prompt_embeds_2 = pipe.text_encoder_2(input_ids_2).last_hidden_state
                pooled_prompt_embeds = pipe.text_encoder_2(input_ids_2).text_embeds

            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device="cuda").long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            model_pred = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids.repeat(latents.shape[0], 1)},
                return_dict=False
            )[0]

            loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if loss.item() < best_loss:
                best_loss = loss.item()

        global_step += 1
        if global_step % 40 == 0:
            elapsed = (time.time() - start_time) / 60
            eta = elapsed * (TOTAL_STEPS - global_step) / global_step if global_step > 0 else 0
            print(f"Step {global_step:3d}/{TOTAL_STEPS} | Loss: {loss.item():.4f} | Best: {best_loss:.4f} | ETA: {eta:.1f}m")
            ckpt_path = os.path.join(output_dir, f"aldar_kose_step_{global_step}")
            accelerator.unwrap_model(unet).save_pretrained(ckpt_path)
    if global_step >= TOTAL_STEPS:
        break

final_path = os.path.join(output_dir, "aldar_kose_2d_final")
accelerator.unwrap_model(unet).save_pretrained(final_path)
print(f"Training complete! Saved to {final_path}")
