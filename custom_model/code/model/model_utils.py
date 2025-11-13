import torch
from torchvision import transforms
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from transformers import ViTModel
from transformers import BertTokenizer, BertModel

model_path = "/repo/project_deepfake/project/custom_model/models/checkpoints/multimodal_model.pth"
tokenizer_path = "/repo/project_deepfake/project/custom_model/models/checkpoints/tokenizer"

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# model = SymmetricMultimodalClassifier()
# model.load_state_dict(torch.load("multimodal_model.pth", map_location=device, weights_only=False))
import __main__ as main
main.SymmetricMultimodalClassifier = SymmetricMultimodalClassifier
main.AttentionPool = AttentionPool
main.CrossAttentionBlock = CrossAttentionBlock

model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()
