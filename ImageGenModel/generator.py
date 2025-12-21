from modelscope.pipelines import pipeline

class ImageGenerator:
    def __init__(self, model_id="NewBieAi-lab/NewBie-image-Exp0.1"):
        self.model_id = model_id
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            print("Loading Newbie model on CPU...")
            self.pipe = pipeline(
                task='text-to-image',
                model=self.model_id,
                device='cpu'  # Force CPU
            )
            print("Model loaded.")

    def generate_image(self, prompt, output_path="newbie_sample.png", height=1024, width=1024, num_inference_steps=28):
        if self.pipe is None:
            self.load_model()
        try:
            print(f"Generating image for prompt: '{prompt}'")
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps
            )
            image = result['images'][0]
            image.save(output_path)
            print(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    generator.generate_image("1girl")

#old, uses stable diffusion     
'''import torch
from diffusers import StableDiffusionPipeline

class ImageGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.model_name = model_name
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            print("Loading model...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            self.pipe.safety_checker = None  # Disable safety checker for uncensored generation
            print("Model loaded.")

    def generate_image(self, prompt, output_path="generated_image.png", num_inference_steps=50, guidance_scale=7.5):
        if self.pipe is None:
            self.load_model()
        try:
            print(f"Generating image for prompt: {prompt}")
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            image.save(output_path)
            print(f"Image saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating image: {e}")
            return None'''
            