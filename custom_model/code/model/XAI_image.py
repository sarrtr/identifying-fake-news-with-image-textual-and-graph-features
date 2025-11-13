from lime import lime_image
from skimage.segmentation import slic
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
import torch
import torch.nn as nn

def reshape_transform(tensor, height=14, width=14):
    # Remove CLS token
    result = tensor[:, 1:, :]  
    # Reshape to (B, H, W, C)
    result = result.reshape(result.size(0), height, width, -1)
    # Permute to (B, C, H, W)
    result = result.permute(0, 3, 1, 2)
    return result
# -------------------------
# Wrapper model for GradCAM
# -------------------------
class MultiModalForCAM(nn.Module):
    """
    Обёртка, принимающая на вход только images и возвращающая logits,
    при этом использует фиксированный текст (input_ids, attention_mask).
    Это позволяет grad-cam получить градиенты по слоям image_encoder,
    но считать цель (target) относительно финального logit'а мульти-модальной модели.
    """
    def __init__(self, full_model, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.full_model = full_model
        # make sure fixed inputs are detached and on the correct device when used
        self.register_buffer('fixed_input_ids', fixed_input_ids.squeeze(0))
        self.register_buffer('fixed_attention_mask', fixed_attention_mask.squeeze(0))

    def forward(self, images):
        # images: tensor (B, C, H, W)
        # repeat fixed text for batch
        batch_size = images.shape[0]
        input_ids = self.fixed_input_ids.unsqueeze(0).repeat(batch_size, 1)
        attention_mask = self.fixed_attention_mask.unsqueeze(0).repeat(batch_size, 1)
        logits, *_ = self.full_model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        return logits

# -------------------------
# LIME predict wrapper for images (multimodal)
# -------------------------
def make_lime_predict_fn(model, input_ids, attention_mask, transform_val, device):
    """
    Вернёт функцию pred_fn(images_np) -> probs_numpy (N x num_classes).
    images_np: array (N, H, W, 3), dtype uint8 (0..255) (как LIME выдаёт).
    Использует transform_val (PIL->tensor->normalize) и фиксированный текст.
    """
    def pred_fn(images_np):
        model.eval()
        tensors = []
        for im in images_np:
            # im: HWC, uint8
            pil = Image.fromarray(im.astype('uint8'), 'RGB')
            x = transform_val(pil).unsqueeze(0)  # 1, C, H, W
            tensors.append(x)
        batch = torch.cat(tensors, dim=0).to(device)  # (N, C, H, W)
        # prepare text inputs (repeat)
        b = batch.shape[0]
        ids = input_ids.unsqueeze(0).repeat(b,1).to(device)
        mask = attention_mask.unsqueeze(0).repeat(b,1).to(device)
        with torch.no_grad():
            logits, *_ = model(ids, mask, batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    return pred_fn

# -------------------------
# Main explain function: runs GradCAM + LIME and plots/saves results
# -------------------------
def explain_image_with_lime_and_gradcam(model, tokenizer,
                                        input_ids, attention_mask, image_tensor,
                                        transform_val,
                                        device='cuda',
                                        lime_samples=50,
                                        top_label=None,
                                        save_path=None):
    """
    input_ids, attention_mask: tensors for the text (1, seq_len) or (seq_len,)
    image_tensor: (C, H, W) tensor (unnormalized? assume already normalized as used by model)
    transform_val: pipeline used in validation (PIL->Tensor->Normalize) for LIME preprocessing
    lime_samples: number of perturbed samples for LIME
    top_label: int or None (if None, use predicted label)
    save_path: если указан, сохраняет картинку
    Возвращает matplotlib.Figure
    """
    model.eval()
    image = image_tensor.clone().detach().to(device)
    # prepare original image in HWC 0..1 for show_cam_on_image
    # undo normalization if transform_val uses Normalize with ImageNet mean/std
    # We'll try to reconstruct 0..1 image from normalized tensor:
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    im_np = image.cpu().numpy().transpose(1,2,0)  # HWC but normalized
    im_01 = (im_np * std + mean).clip(0,1)  # approximate original normalized to 0..1

    # --- GradCAM ---
    # determine a target layer inside ViT
    # target_layer = get_vit_target_layer(model.image_encoder)
    target_layer = model.image_encoder.encoder.layer[-1].output
    # wrap multimodal model for GradCAM
    wrapper = MultiModalForCAM(model, input_ids.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device))
    wrapper.to(device)
    # create GradCAM object
    cam = GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape_transform)
    # input to CAM must be a tensor (B, C, H, W)
    with torch.no_grad():
        inp = image.unsqueeze(0).to(device)
    # optionally compute target category: top_label or predicted
    with torch.no_grad():
        logits, *_ = model(input_ids=input_ids.unsqueeze(0).to(device),
                           attention_mask=attention_mask.unsqueeze(0).to(device),
                           images=inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_label = int(probs.argmax(axis=1)[0])
    target_category = pred_label if top_label is None else int(top_label)

    grayscale_cam = cam(input_tensor=inp, targets=None)[0]  # HxW
    cam_image = show_cam_on_image(im_01, grayscale_cam, use_rgb=True)  # uint8 HxW3

    # --- LIME (image) ---
    explainer = lime_image.LimeImageExplainer()
    predict_fn = make_lime_predict_fn(model, input_ids, attention_mask, transform_val, device)
    # lime wants HWC uint8 0..255
    im_for_lime = (im_01 * 255).astype('uint8')
    # run explain_instance (this is relatively heavy)
    explanation = explainer.explain_instance(
        im_for_lime,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=lime_samples,          
        segmentation_fn=lambda x: slic(x, n_segments=25, compactness=5)  # faster/better segmentation
    )
    # choose label to visualize
    label_to_vis = target_category
    temp, mask = explanation.get_image_and_mask(label_to_vis,
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    # temp: HxWx3 uint8 with highlighted superpixels
    lime_overlay = temp

    # --- plot side-by-side ---
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow((im_01.clip(0,1)))
    axes[0].set_title('Original (approx)')
    axes[0].axis('off')

    axes[1].imshow(cam_image)
    axes[1].set_title(f'Grad-CAM (pred={pred_label})')
    axes[1].axis('off')

    axes[2].imshow(lime_overlay)
    axes[2].set_title('LIME overlay')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        
    return fig, {'pred_prob': probs[0].tolist(), 'pred_label': pred_label, 'grayscale_cam': grayscale_cam, 'lime_mask': mask}
