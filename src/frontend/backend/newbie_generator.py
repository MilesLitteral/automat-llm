# Optional image generator
try:
    import torch
    from diffusers import NewbiePipeline

    class NewbieGenerator:
        def __init__(self):
            model_id = "NewBie-AI/NewBie-image-Exp0.1"
            self.pipe = NewbiePipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).to("cuda")  # fallback: .to("cpu") if no GPU

        def generate_image(self, prompt, filename):
            image = self.pipe(
                prompt,
                height=1024,
                width=1024,
                num_inference_steps=28,
            ).images[0]
            image.save(filename)
            print(f"Saved to {filename}")

except Exception as e:
    print(f"⚠️ Image generator not loaded: {e}")
    # Fallback dummy generator
    class DummyGenerator:
        def generate_image(self, prompt, filename):
            print(f"⚠️ (Dummy) Skipped generating image for '{prompt}' → {filename}")
    generator = DummyGenerator()