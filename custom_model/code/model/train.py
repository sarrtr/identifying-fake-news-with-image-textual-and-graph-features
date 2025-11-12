import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F
from transformers import ViTModel
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MultimodalNewsDataset(Dataset):
    def __init__(self, txt_file, image_dir, tokenizer, max_len=128, transform=None):
        self.data = pd.read_csv(txt_file, sep='\t')
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
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
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
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
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset_path = '/repo/project_deepfake/project/fakeddit_dataset'
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = MultimodalNewsDataset(
    txt_file=dataset_path+'/text/text.txt',
    image_dir=dataset_path+'/images',
    tokenizer=tokenizer,
    transform=transform_train 
)

# Split indices
train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=dataset.data['2_way_label']
)

# Train/val subsets
# small_train_idx = train_idx[:5000]
# small_val_idx   = val_idx[:2000]

train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx)

# Override val transform manually
val_set.dataset.transform = transform_val  

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

# -----------------------
# Cross-Attention Block
# -----------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_k, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_k = nn.LayerNorm(dim_k)

    def forward(self, q, k, v, key_padding_mask=None):
        q_norm, k_norm = self.norm_q(q), self.norm_k(k)
        attn_out, attn_weights = self.attn(q_norm, k_norm, v, key_padding_mask=key_padding_mask)
        return q + attn_out, attn_weights


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        returns: pooled (batch, hidden_dim)
        """
        weights = self.proj(x).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(weights, dim=-1)  # attention over sequence
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return pooled
    
# -----------------------
# Symmetric Multimodal Classifier
# -----------------------
class SymmetricMultimodalClassifier(nn.Module):
    def __init__(self, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224',
                 hidden_dim=768, num_classes=2):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.image_encoder = ViTModel.from_pretrained(image_model)

        self.text_to_img = CrossAttentionBlock(hidden_dim, hidden_dim)
        self.img_to_text = CrossAttentionBlock(hidden_dim, hidden_dim)

        self.text_pool = AttentionPool(hidden_dim)
        self.img_pool  = AttentionPool(hidden_dim)

        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, images):
        # Encode
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        img_feat = self.image_encoder(pixel_values=images).last_hidden_state

        # --- Symmetric Cross-Attention ---
        text_key_padding = (input_ids == tokenizer.pad_token_id)  # True for PAD

        # text -> image cross-attention
        text_cross, attn_t2i = self.text_to_img(text_feat, img_feat, img_feat)  # no mask needed for image
        # image -> text cross-attention
        img_cross, attn_i2t = self.img_to_text(img_feat, text_cross, text_cross,
                                            key_padding_mask=text_key_padding)

        # --- Mean pooling ---
        # text_emb = text_cross.mean(dim=1)
        # img_emb = img_cross.mean(dim=1)

        text_emb = self.text_pool(text_cross)
        img_emb = self.img_pool(img_cross)
        
        # --- Normalized embeddings for contrastive learning ---
        text_emb_n = F.normalize(self.text_proj(text_emb), dim=-1)
        img_emb_n = F.normalize(self.img_proj(img_emb), dim=-1)

        # --- Classification logits (use cross-attended text embedding) ---
        logits = self.classifier(text_emb)

        return logits, text_emb_n, img_emb_n, attn_t2i, attn_i2t

# -----------------------
# Symmetric InfoNCE Contrastive Loss
# -----------------------
def contrastive_loss(text_emb, img_emb, temperature=0.05):
    sim_matrix = torch.matmul(text_emb, img_emb.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss_i = F.cross_entropy(sim_matrix, labels)
    loss_t = F.cross_entropy(sim_matrix.T, labels)
    return (loss_i + loss_t) / 2

import matplotlib.pyplot as plt

# -----------------------
# Pretraining: Cross-Attention with InfoNCE
# -----------------------
def pretrain_contrastive(model, train_loader, optimizer, device='cuda', epochs=5):
    model.train()
    history = {"epoch": [], "loss": []}
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [train contrastive]"):
            input_ids, attention_mask, images = (
                input_ids.to(device), attention_mask.to(device), images.to(device)
            )
            _, t_emb, i_emb, _, _ = model(input_ids, attention_mask, images)
            loss = contrastive_loss(t_emb, i_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        print(f"[Pretrain] Epoch {epoch+1}: Loss={avg_loss:.4f}")

    return history


# -----------------------
# Fine-tuning: Classification
# -----------------------
def finetune_classification(model, train_loader, val_loader, optimizer, device='cuda', epochs=3):
    ce_loss = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, total_acc = 0, 0
        for input_ids, attention_mask, images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            input_ids, attention_mask, images, labels = (
                input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)
            )
            logits, t_emb, i_emb, _, _ = model(input_ids, attention_mask, images)
            loss_cls = ce_loss(logits, labels)
            loss_contrast = contrastive_loss(t_emb, i_emb)
            loss = loss_cls + 0.1 * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            total_loss += loss.item()
            total_acc += acc

        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                input_ids, attention_mask, images, labels = (
                    input_ids.to(device), attention_mask.to(device), images.to(device), labels.to(device)
                )
                logits, t_emb, i_emb, _, _ = model(input_ids, attention_mask, images)
                loss_cls = ce_loss(logits, labels)
                loss_contrast = contrastive_loss(t_emb, i_emb)
                loss = loss_cls + 0.1 * loss_contrast

                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()
                val_loss += loss.item()
                val_acc += acc

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return history

model = SymmetricMultimodalClassifier().to(device)
# Freeze BERT and ViT initially
for p in model.text_encoder.parameters():
    p.requires_grad = False
for p in model.image_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW([
    {"params": model.text_to_img.parameters(), "lr": 2e-4},
    {"params": model.img_to_text.parameters(), "lr": 2e-4},
    {"params": model.text_proj.parameters(), "lr": 2e-4},
    {"params": model.img_proj.parameters(), "lr": 2e-4},
], weight_decay=0.01)


# Step 1: Pretrain cross-attention
history_step1 = pretrain_contrastive(model, train_loader, optimizer, device=device, epochs=5)

# Step 2: Fine-tune classifier
history_step2 = finetune_classification(model, train_loader, val_loader, optimizer, device=device, epochs=3)
# After a few epochs, unfreeze top layers of text/image encoders
for name, p in model.text_encoder.named_parameters():
    if 'encoder.layer.11' in name or 'pooler' in name:
        p.requires_grad = True

for name, p in model.image_encoder.named_parameters():
    if 'encoder.layer.11' in name:
        p.requires_grad = True


optimizer = torch.optim.AdamW([
    {"params": model.text_encoder.parameters(), "lr": 1e-6},   # frozen initially, small LR in case you unfreeze later
    {"params": model.image_encoder.parameters(), "lr": 1e-6},  # same here
    {"params": model.text_to_img.parameters(), "lr": 2e-4},    # trainable layers
    {"params": model.img_to_text.parameters(), "lr": 2e-4},
    {"params": model.text_proj.parameters(), "lr": 2e-4},
    {"params": model.img_proj.parameters(), "lr": 2e-4},
    {"params": model.classifier.parameters(), "lr": 2e-4},
], weight_decay=0.01)

history_step3 = finetune_classification(model, train_loader, val_loader, optimizer, device=device, epochs=3)

tokenizer.save_pretrained("checkpoints/tokenizer/")

save_path = "checkpoints/multimodal_model.pth"

torch.save(model, save_path)

print(f"Model saved to {save_path}")


# --- Plot contrastive pretraining loss ---
plt.figure(figsize=(6, 4))
plt.plot(history_step1["epoch"], history_step1["loss"], marker='o', color='royalblue')
plt.title("Contrastive Pretraining Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig('pretrain_contrastive_loss.png')
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(history_step2["epoch"], history_step2["train_loss"], marker='o', label="Train Loss")
axs[0].plot(history_step2["epoch"], history_step2["val_loss"], marker='o', label="Val Loss")
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(history_step2["epoch"], history_step2["train_acc"], marker='o', label="Train Acc")
axs[1].plot(history_step2["epoch"], history_step2["val_acc"], marker='o', label="Val Acc")
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
axs[1].grid(True)
plt.savefig('finetuned_unfreeze_last_layers_encoders.png')
plt.show()

# --- Plot training curves ---
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(history_step3["epoch"], history_step3["train_loss"], marker='o', label="Train Loss")
axs[0].plot(history_step3["epoch"], history_step3["val_loss"], marker='o', label="Val Loss")
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(history_step3["epoch"], history_step3["train_acc"], marker='o', label="Train Acc")
axs[1].plot(history_step3["epoch"], history_step3["val_acc"], marker='o', label="Val Acc")
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
axs[1].grid(True)
plt.savefig('finetuned_freeze_encoders.png')
plt.show()