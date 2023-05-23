from diffusers import StableDiffusionPipeline

TOKEN = "YOUR_AUTH_TOKEN"

# get your token at https://huggingface.co/settings/tokens
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    use_auth_token=TOKEN,
    cache_dir="/home/user/cache")
pipe.to("cuda")
prompt = "a photograph of an astronaut riding a horse"
image = pipe(prompt)["sample"][0]
image.save("image.png")