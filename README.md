# text-to-image
!pip install -qq "ipywidgets>=7,<8"
!git clone https://github.com/huggingface/diffusers
%cd /content/diffusers
!pip install .
%cd /content/diffusers/examples/dreambooth
!pip install -r requirements.txt
!pip install wandb
!pip install bitsandbytes
!pip install transformers gradio ftfy accelerate
!pip install xformers
# Data Understranding
import os
%cd /content

if os.path.exists("/content/custom_dataset"):
    print("Removing existing custom_dataset folder")
    !rm -rf /content/custom_dataset

print("Creating new custom_dataset folder")
!mkdir /content/custom_dataset
!mkdir /content/custom_dataset/class_images
!mkdir /content/custom_dataset/instance_images

print('Custom Dataset folder is created: /content/custom_dataset')
# Image Resize
from PIL import Image
import os
import IPython.display as display
import matplotlib.pyplot as plt

def resize_and_crop_images(folder_path, target_size=512):
    """
    Resize the images in a folder to have a smaller edge of the specified target size and display them.

    Parameters:
    - folder_path (str): Path to the folder containing the images.
    - target_size (int): Desired size for the smaller edge (default is 512).
    """
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Open the image
            image = Image.open(file_path)

            # Get the original width and height
            width, height = image.size

            # Calculate the new size while maintaining the aspect ratio
            if width <= height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_width = int(width * (target_size / height))
                new_height = target_size

            # Resize the image
            resized_image = image.resize((new_width, new_height))

            left = (new_width - target_size) // 2
            top = (new_height - target_size) // 2
            right = (new_width + target_size) // 2
            bottom = (new_height + target_size) // 2

            # Perform the center crop
            cropped_image = resized_image.crop((left, top, right, bottom))
            cropped_image.save(file_path)
  def show_images_in_one_row(folder_path, target_size=256):
    images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = Image.open(file_path)
            img = img.resize((target_size, int(target_size * img.size[1] / img.size[0])))
            images.append(img)

    # Display images in one row
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 3, 3))
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()
    # Class Images
folder_path = '/content/custom_dataset/class_images'
if len(os.listdir(folder_path)):
  resize_and_crop_images(folder_path)
  show_images_in_one_row(folder_path)
  # Instance Images
folder_path = '/content/custom_dataset/instance_images'
resize_and_crop_images(folder_path)
show_images_in_one_row(folder_path)
# Train Image
%cd /content/diffusers/examples/dreambooth
!python train_dreambooth.py --pretrained_model_name_or_path 'runwayml/stable-diffusion-v1-5' \
                            --revision "fp16" \
                            --instance_data_dir '/content/custom_dataset/instance_images' \
                            --class_data_dir '/content/custom_dataset/class_images' \
                            --instance_prompt 'A photo of a shan girl ' \
                            --class_prompt 'A photo of a girl' \
                            --with_prior_preservation \
                            --prior_loss_weight 1.0 \
                            --num_class_images 100 \
                            --output_dir '/content/outputs' \
                            --resolution 512 \
                            --train_text_encoder \
                            --train_batch_size 5 \
                            --sample_batch_size 5 \
                            --max_train_steps 600 \
                            --checkpointing_steps 200 \
                            --gradient_accumulation_steps 1 \
                            --gradient_checkpointing \
                            --learning_rate 5e-6 \
                            --lr_scheduler 'constant' \
                            --lr_warmup_steps=0 \
                            --use_8bit_adam \
                            --validation_prompt 'A photo of a shan girl' \
                            --num_validation_images 4 \
                            --mixed_precision "fp16" \
                            --enable_xformers_memory_efficient_attention \
                            --set_grads_to_none \
                            --report_to 'wandb'
# modeling
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os

trained_model_path = '/content/outputs'

unet = UNet2DConditionModel.from_pretrained(trained_model_path + '/unet')
text_encoder = CLIPTextModel.from_pretrained(trained_model_path + '/text_encoder')

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet,
    text_encoder=text_encoder, dtype=torch.float16,
).to("cuda")
# User Interface
import gradio as gr

def inference(prompt, num_samples, negative_prompt, guidance_scale,
              num_inference_steps, height, width):
    all_images = []
    images = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images
    all_images.extend(images)
    return all_images

with gr.Blocks() as demo:
    gr.HTML("<h2 style=\"font-size: 2em; font-weight: bold\" align=\"center\">MOSIAS TEAM</h2>")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")
            samples = gr.Slider(label="Samples", value=1, maximum=12)
            guidance_scale = gr.Slider(label="Guidance Scale", value=7.5, maximum=30)
            num_inference_steps = gr.Slider(label="Inference Steps", value=50, maximum=500)
            height = gr.Slider(label="Height", value=512)
            width = gr.Slider(label="Width", value=512)
            run = gr.Button(value="Run")
        with gr.Column():
            gallery = gr.Gallery(show_label=False)

    run.click(inference, inputs=[prompt, samples, negative_prompt, guidance_scale, num_inference_steps, height, width], outputs=gallery)
    gr.Examples([["A photo of a shan girl", 1, "", 7.5, 150, 512, 512]], [prompt, samples, negative_prompt, guidance_scale, num_inference_steps, height, width], gallery, inference, cache_examples=False)

demo.launch(debug=True)
