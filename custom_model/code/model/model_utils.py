import torch
from torchvision import transforms
from transformers import BertTokenizer
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from transformers import ViTModel
from transformers import BertTokenizer, BertModel

model_path = "models/checkpoints/multimodal_model.pth"
tokenizer_path = "models/checkpoints/tokenizer"

class CrossAttentionBlock(nn.Module):
    """
    Implements cross-modal attention mechanism between text and image features
    Uses multi-head attention with layer normalization and residual connections
    """
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
    """
    Implements attention-based pooling for sequence features
    Learns importance weights for each sequence element and computes weighted average
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = self.proj(x).squeeze(-1)
        weights = F.softmax(weights, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled

class SymmetricMultimodalClassifier(nn.Module):
    """
    Main multimodal classification model with symmetric cross-attention
    Combines BERT for text processing and ViT for image processing
    Uses bidirectional cross-attention between modalities for feature fusion
    """
    def __init__(self, text_model='bert-base-uncased', image_model='google/vit-base-patch16-224',
                 hidden_dim=768, num_classes=2):
        super().__init__()
        # Initialize pre-trained text and image encoders
        self.text_encoder = BertModel.from_pretrained(text_model)
        self.image_encoder = ViTModel.from_pretrained(image_model)

        # Cross-attention blocks for modality interaction
        self.text_to_img = CrossAttentionBlock(hidden_dim, hidden_dim)
        self.img_to_text = CrossAttentionBlock(hidden_dim, hidden_dim)

        # Attention pooling layers for sequence aggregation
        self.text_pool = AttentionPool(hidden_dim)
        self.img_pool  = AttentionPool(hidden_dim)

        # Projection layers for normalized embeddings
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.img_proj = nn.Linear(hidden_dim, hidden_dim)

        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask, images):
        # Encode text and image inputs using pre-trained models
        text_feat = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        img_feat = self.image_encoder(pixel_values=images).last_hidden_state

        # Create padding mask for text tokens
        text_key_padding = (input_ids == tokenizer.pad_token_id)

        # Apply bidirectional cross-attention between modalities
        text_cross, attn_t2i = self.text_to_img(text_feat, img_feat, img_feat)
        img_cross, attn_i2t = self.img_to_text(img_feat, text_cross, text_cross,
                                            key_padding_mask=text_key_padding)

        # Pool cross-attended features using attention pooling
        text_emb = self.text_pool(text_cross)
        img_emb = self.img_pool(img_cross)
        
        # Create normalized embeddings for contrastive learning
        text_emb_n = F.normalize(self.text_proj(text_emb), dim=-1)
        img_emb_n = F.normalize(self.img_proj(img_emb), dim=-1)

        # Generate final classification logits from text embeddings
        logits = self.classifier(text_emb)

        return logits, text_emb_n, img_emb_n, attn_t2i, attn_i2t

# Configure device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained tokenizer for text processing
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Define image transformation pipeline for model input
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Register custom classes for safe model loading
import __main__ as main
main.SymmetricMultimodalClassifier = SymmetricMultimodalClassifier
main.AttentionPool = AttentionPool
main.CrossAttentionBlock = CrossAttentionBlock

# Add custom classes to PyTorch's safe globals for deserialization
torch.serialization.add_safe_globals([
    SymmetricMultimodalClassifier,
    AttentionPool,
    CrossAttentionBlock
])

# Load pre-trained model weights and move to appropriate device
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()