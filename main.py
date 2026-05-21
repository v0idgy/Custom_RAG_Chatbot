from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Prompt input from user
prompt = input("enter the prompt here: ")

# Load the pretrained Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Generate the image using the user's prompt
with torch.autocast("cuda"):
    image = pipe(prompt).images[0]

# Save the image to a file
image.save("results/generated_image.png")
print("Image saved as 'results/generated_image.png'.")

# Display the image
plt.imshow(image)
plt.axis("off")
plt.title(f"Generated for prompt:\n{prompt}")
plt.show()
