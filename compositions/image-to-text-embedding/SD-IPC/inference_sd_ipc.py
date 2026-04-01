import argparse
import os, sys
from glob import glob
from tqdm import tqdm

from PIL import Image

import torch

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPVisionModel, CLIPImageProcessor
from prompt_clip import CLIPVisionModelWithPrompt

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

@torch.no_grad()
def main(args):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_path, 
        torch_dtype=torch.float32,
    ).to(device)
    tokenizer = pipe.tokenizer

    clip_for_inverse_matrix = CLIPModel.from_pretrained(args.clip_path).to(device)
    image_processer = CLIPImageProcessor.from_pretrained(args.clip_path)
    inv_text = torch.linalg.pinv(clip_for_inverse_matrix.text_projection.weight, atol=0.3)
    visual_projection = clip_for_inverse_matrix.visual_projection.weight

    if not os.path.exists(os.path.join(args.pretrained_path, "clip")):
        clip = clip_for_inverse_matrix.vision_model
    else:
        clip = CLIPVisionModelWithPrompt.from_pretrained(os.path.join(args.pretrained_path, "clip"), prompt_length=args.clip_prompt_length).to(device)
        del clip_for_inverse_matrix
    
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.jpg")))
    else:
        l_img_paths = [args.input_image]

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = tokenizer(
        args.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    text_input = inputs.input_ids.to(device)
    text_masks = inputs.attention_mask.to(device)
    text_embeddings = pipe.text_encoder(text_input)[0]

    for img_path in tqdm(l_img_paths):
        bname = os.path.basename(img_path).split(".")[0]
        image = Image.open(img_path).convert("RGB")

        clip_image = image_processer(image, return_tensors='pt').pixel_values.to(device)
        image_emb = clip(pixel_values=clip_image,)
        image_emb = image_emb.pooler_output

        # SD-IPC convert
        image_emb_proj = image_emb @ visual_projection.T 
        image_emb_proj = image_emb_proj @ inv_text.T
        image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=1, keepdim=True)
        image_emb_proj = 27.5 * image_emb_proj

        convert_text_embeddings = torch.zeros_like(text_embeddings)
        convert_text_embeddings[:, 0] = text_embeddings[:, 0]
        convert_text_embeddings[:, 1:] = image_emb_proj.unsqueeze(1)

        convert_edit_embeddings  = text_embeddings.clone()
        convert_edit_embeddings[:, text_masks.sum(1)[0]-1:] = image_emb_proj.unsqueeze(1) + args.alpha * text_embeddings[:, text_masks.sum(1)[0]-1:]

        if args.onlyprompt:
            prompt_embeds = text_embeddings
        elif args.edit:
            prompt_embeds = convert_edit_embeddings
        else:
            prompt_embeds = convert_text_embeddings

        image = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        ).images[0]
        image.save(args.output_dir + args.save_prefix + bname + ".jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='./example_images/')
    parser.add_argument('--prompt', type=str, default="")
    parser.add_argument("--clip_prompt_length", type=int, default=50)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--onlyprompt", action="store_true", default=False)
    parser.add_argument("--edit", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default='./stable-diffusion-v1-4/',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--clip_path",
        type=str,
        default='./clip-vit-large-patch14/',
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=10, help="A seed for reproducible training.")

    args = parser.parse_args()

    main(args)
