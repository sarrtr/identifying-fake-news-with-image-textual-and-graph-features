import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import Blip2Processor
from tqdm import tqdm

class TxtImageTextDataset(Dataset):
    def __init__(self, txt_file, image_folder, processor, image_size=(224, 224)):
        """
        Args:
            txt_file: path to .txt with columns: image_name \t text \t label
            image_folder: folder containing .jpg images
            processor: HuggingFace processor (BLIP2Processor)
            image_size: tuple for resizing images
        """
        self.image_folder = image_folder
        self.processor = processor
        self.image_size = image_size

        # Read samples
        self.samples = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                img_name, text, label = parts[0], parts[1], float(parts[2])
                self.samples.append((img_name, text, label))

        self.labels = torch.tensor([s[2] for s in self.samples], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text, label = self.samples[idx]
        img_path = os.path.join(self.image_folder, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size)

        # Use processor to tokenize/process text and prepare image tensor
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs['pixel_values'], inputs['input_ids'], label
