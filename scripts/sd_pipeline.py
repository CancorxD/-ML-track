"""
Stable Diffusion helper. Loads the pipeline lazily and provides generate_frame().
"""
from typing import Optional
import torch
from diffusers import StableDiffusionPipeline


_PIPELINE: Optional[StableDiffusionPipeline] = None




def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"




def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5") -> StableDiffusionPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE


    device = _device()
    torch_dtype = torch.float16 if device == "cuda" else torch.float32


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)


    _PIPELINE = pipe
    return pipe




def generate_frame(prompt: str, seed: int = 42, steps: int = 50):
    pipe = load_pipeline()
    device = _device()


    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe(prompt, num_inference_steps=steps, generator=gen, guidance_scale=7.5)
    return out.images[0]