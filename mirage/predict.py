# usage: python predict.py --image my_data/image/news1.jpg --text_file my_data/text/news1.txt

import argparse
import os
import torch
from PIL import Image
import clip
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from data import load_config
from models import get_model
from utils import load_model_checkpoint

def load_image_embedding(image_path, device):

    #cached model
    # processor = Blip2Processor.from_pretrained("C:/Users/kuzzm/.cache/huggingface/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28")
    # vision = Blip2ForConditionalGeneration.from_pretrained("C:/Users/kuzzm/.cache/huggingface/hub/models--Salesforce--blip2-flan-t5-xl/snapshots/0eb0d3b46c14c1f8c7680bca2693baafdb90bb28").vision_model.to(device)
    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    vision = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").vision_model.to(device)
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(img).data['pixel_values'][0]  # numpy-like or tensor
    x = torch.Tensor(pixel_values).unsqueeze(0).to(device)  # shape [1, C, H, W]
    with torch.no_grad():
        emb = vision(x).pooler_output  # [1, hidden]
    return emb  # tensor on device

def load_text_embedding(text, device):
    clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)  # [1, hidden]
    return emb

def get_model_from_config(mode, model_class, device):
    cfg_path = f"configs/{mode}/{model_class}.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    model = get_model(cfg).to(device)
    model, _ = load_model_checkpoint(model, cfg['training']['save_path'])
    return model

def to_prob(tensor):
    t = tensor.detach().cpu()
    if t.min() >= 0.0 and t.max() <= 1.0:
        return t
    else:
        return torch.sigmoid(t)

def predict_single(image_path, text, device, image_model_class="linear", text_model_class="linear"):
    # 1) compute embeddings
    img_emb = load_image_embedding(image_path, device)   # [1, D_img]
    txt_emb = load_text_embedding(text, device)         # [1, D_txt]

    # 2) load models (from configs)
    print("Loading image model...")
    image_model = get_model_from_config("image", image_model_class, device)
    print("Loading text model...")
    text_model = get_model_from_config("text", text_model_class, device)

    # 3) forward through models
    image_model.eval()
    text_model.eval()
    with torch.no_grad():
        # Many repo models expect float tensors on device; adjust if needed
        img_out = image_model(img_emb.float().to(device))
        txt_out = text_model(txt_emb.float().to(device))

    # 4) convert to probabilities and scalars
    p_img = to_prob(img_out).squeeze().item()
    p_txt = to_prob(txt_out).squeeze().item()

    # 5) combine probabilities (simple average)
    p_combined = float((p_img + p_txt) / 2.0)

    label = "fake" if p_combined > 0.5 else "real"
    return {
        "prob_image": float(p_img),
        "prob_text": float(p_txt),
        "prob_combined": float(p_combined),
        "label": label
    }

def main():
    parser = argparse.ArgumentParser(description="Predict whether a news pair (image + text) is fake using trained MiRAGe models.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file (jpg/png).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Caption/text string to evaluate (wrap in quotes).")
    group.add_argument("--text_file", type=str, help="Path to txt file containing the caption on one line.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--image_model_class", type=str, default="linear", help="Which image model class to use (linear, cbm-encoder, ...).")
    parser.add_argument("--text_model_class", type=str, default="linear", help="Which text model class to use (linear, tbm-encoder, ...).")
    args = parser.parse_args()

    if args.text_file:
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(args.text_file)
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = args.text

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)

    device = args.device

    print("Running prediction...")
    out = predict_single(args.image, text, device, args.image_model_class, args.text_model_class)

    print("=== Prediction result ===")
    print(f"Image probability (fake): {out['prob_image']:.4f}")
    print(f"Text  probability (fake): {out['prob_text']:.4f}")
    print(f"Combined probability (fake): {out['prob_combined']:.4f}")
    print(f"Final label: {out['label']}")
    print("=========================")

if __name__ == "__main__":
    main()
