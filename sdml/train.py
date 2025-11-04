import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel, ViTModel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.vit import VisionTransformer

from tqdm import tqdm

class MultimodalNewsDataset(Dataset):
    def __init__(self, txt_file, image_dir, tokenizer, max_len=32, transform=None):
        self.data = pd.read_csv(txt_file, sep='\t')
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['clean_title'])
        img_id = str(row['id'])
        label = torch.tensor(int(row['2_way_label']), dtype=torch.long)
        
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(f"[WARN] Ошибка при открытии {img_path}: {e}")
            # Возвращаем пустое изображение, чтобы не ломать batch
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        
        image = self.transform(image)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask, image, label


class CrossAttentionFusion(nn.Module):
    """A lightweight cross-attention block between text and image embeddings."""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.cross_attn_text = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_img = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.norm_i = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        # text_feat: (B, T, D)
        # img_feat: (B, P, D)
        # text attends to image and image attends to text
        t2i, _ = self.cross_attn_text(query=text_feat, key=img_feat, value=img_feat)
        i2t, _ = self.cross_attn_img(query=img_feat, key=text_feat, value=text_feat)

        text_out = self.norm_t(text_feat + t2i)
        img_out = self.norm_i(img_feat + i2t)
        return text_out, img_out
        

class SDMLMultimodal(nn.Module):
    def __init__(self, 
                 text_model='bert-base-uncased', 
                 vision_model='google/vit-base-patch16-224-in21k',
                 num_classes=2,
                 freeze_pretrained=True):
        super().__init__()
        
        # --- Text Encoder ---
        self.text_encoder = BertModel.from_pretrained(text_model)
        text_embed_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_embed_dim, 512)
        
        # --- Vision Encoder ---
        self.vision_encoder = ViTModel.from_pretrained(vision_model)
        img_embed_dim = self.vision_encoder.config.hidden_size
        self.image_proj = nn.Linear(img_embed_dim, 512)

        # --- Cross-Attention Fusion ---
        self.fusion = CrossAttentionFusion(embed_dim=512, num_heads=8)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        # --- Optionally freeze pretrained encoders ---
        if freeze_pretrained:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.vision_encoder.parameters():
                param.requires_grad = False


    def forward(self, input_ids, attention_mask, images):
        # --- Text branch ---
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = text_out.last_hidden_state  # (B, T, hidden)
        text_tokens = self.text_proj(text_tokens)

        # --- Vision branch ---
        img_out = self.vision_encoder(images)
        img_tokens = img_out.last_hidden_state  # (B, P, hidden)
        img_tokens = self.image_proj(img_tokens)

        # --- Cross-attention fusion ---
        text_fused, img_fused = self.fusion(text_tokens, img_tokens)

        # Take CLS (first token) representations from both branches
        text_cls = text_fused[:, 0, :]
        img_cls = img_fused[:, 0, :]

        fused = torch.cat([text_cls, img_cls], dim=-1)
        logits = self.classifier(fused)
        return logits



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for input_ids, attention_mask, images, labels in progress_bar:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / ((progress_bar.n + 1) * loader.batch_size):.2f}%'
        })

    avg_loss = total_loss / len(loader)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0

    progress_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for input_ids, attention_mask, images, labels in progress_bar:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            images, labels = images.to(device), labels.to(device)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / ((progress_bar.n + 1) * loader.batch_size):.2f}%'
            })

    avg_loss = total_loss / len(loader)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    dataset_path = '/repo/project_deepfake/project/fakeddit_dataset'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = MultimodalNewsDataset(dataset_path+'/text/text.txt', dataset_path+'/images', tokenizer)
    indices = list(range(len(dataset)))
    print(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=dataset.data['2_way_label'])

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    model = SDMLMultimodal(freeze_pretrained=True).to(device)


    # checkpoint_path = "sdml_multimodal.pt"
    # state_dict = torch.load(checkpoint_path, map_location=device)

    # model.load_state_dict(state_dict)


    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # only trainable params
        lr=1e-4,
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []


    for epoch in range(3):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"[Step 1] Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Step 2: Unfreeze text encoder only 
    for param in model.text_encoder.parameters():
        param.requires_grad = True

    # Optimizer: smaller LR for encoder, higher LR for projection/classifier
    optimizer = torch.optim.AdamW([
        {'params': model.text_encoder.parameters(), 'lr': 1e-6},
        {'params': list(model.text_proj.parameters()) + list(model.image_proj.parameters()) + list(model.classifier.parameters()), 'lr': 2e-4}
    ])

    for epoch in range(2):  # fine-tune text encoder 1–2 epochs
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"[Step 2] Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # Step 3: Unfreeze vision encoder 
    for param in model.vision_encoder.parameters():
        param.requires_grad = True

    # Optimizer: fine-tune both encoders + projection/classifier
    optimizer = torch.optim.AdamW([
        {'params': model.text_encoder.parameters(), 'lr': 1e-7},  # smaller
        {'params': model.vision_encoder.parameters(), 'lr': 1e-6}, 
        {'params': list(model.text_proj.parameters()) + list(model.image_proj.parameters()) + list(model.classifier.parameters()), 'lr': 2e-4}
    ])


    # Continue fine-tuning for remaining epochs
    for epoch in range(5): 
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"[Step 3] Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        torch.save(model.state_dict(), f"sdml_multimodal_{epoch}.pt")
        print("Model saved to sdml_multimodal_{epoch}.pt")

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-o', label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=150)