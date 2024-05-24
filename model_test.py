import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "/share/program/dxs/RSISR/pretrain_weights", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
print(pipe.type())
url = "https://hf-mirror.com/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

# init_image = load_image(url).convert("RGB")
# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt, image=init_image).images
