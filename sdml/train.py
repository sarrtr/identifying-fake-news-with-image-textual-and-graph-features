import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.vit import VisionTransformer

class MultimodalNewsDataset(Dataset):
    def __init__(self, txt_file, image_dir, tokenizer, max_len=32, transform=None):
        self.data = pd.read_csv(txt_file, sep='\t')
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
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

class SDMLMultimodal(nn.Module):
    def __init__(self, text_model='bert-base-uncased', num_classes=2, img_embed_dim=768, text_embed_dim=768):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.text_proj = nn.Linear(text_embed_dim, 512)

        self.vision_encoder = VisionTransformer(img_size=224, patch_size=16, embed_dim=img_embed_dim)
        self.image_proj = nn.Linear(img_embed_dim, 512)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, images):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0, :]  # CLS token
        text_feat = self.text_proj(text_cls)
        
        img_out = self.vision_encoder(images)
        img_cls = img_out[:, 0, :]  # CLS token
        img_feat = self.image_proj(img_cls)
        
        fused = torch.cat([text_feat, img_feat], dim=-1)
        logits = self.classifier(fused)
        return logits

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for input_ids, attention_mask, images, labels in loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, images, labels in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            images, labels = images.to(device), labels.to(device)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

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

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = SDMLMultimodal().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    epochs = range(1, len(train_losses) + 1)

    torch.save(model.state_dict(), "sdml_multimodal.pt")
    print("Model saved to sdml_multimodal.pt")


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