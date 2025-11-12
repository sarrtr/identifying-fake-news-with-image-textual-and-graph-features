import torch
from torchvision import transforms
from transformers import BertTokenizer
from train import SymmetricMultimodalClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = SymmetricMultimodalClassifier()
model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
model.to(device)
model.eval()