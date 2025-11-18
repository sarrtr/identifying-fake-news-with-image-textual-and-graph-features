from lime import lime_image
from skimage.segmentation import slic
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn

def reshape_transform(tensor, height=14, width=14):
    """
    Transform ViT encoder output for GradCAM compatibility
    Removes CLS token and reshapes to standard CNN feature map format
    Converts from transformer sequence to spatial dimensions
    """
    # Remove CLS token from transformer output
    result = tensor[:, 1:, :]  
    # Reshape to (B, height, width, channels) format
    result = result.reshape(result.size(0), height, width, -1)
    # Permute to standard PyTorch (B, C, H, W) format
    result = result.permute(0, 3, 1, 2)
    return result

class MultiModalForCAM(nn.Module):
    """
    Wrapper model for GradCAM that fixes text inputs and only varies images
    Enables gradient computation through image encoder while using constant text context
    Essential for visualizing which image regions influence multimodal predictions
    """
    def __init__(self, full_model, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.full_model = full_model
        # Store fixed text inputs as buffers for device compatibility
        self.register_buffer('fixed_input_ids', fixed_input_ids.squeeze(0))
        self.register_buffer('fixed_attention_mask', fixed_attention_mask.squeeze(0))

    def forward(self, images):
        # Repeat fixed text inputs to match image batch size
        batch_size = images.shape[0]
        input_ids = self.fixed_input_ids.unsqueeze(0).repeat(batch_size, 1)
        attention_mask = self.fixed_attention_mask.unsqueeze(0).repeat(batch_size, 1)
        # Forward pass through multimodal model with fixed text and variable images
        logits, *_ = self.full_model(input_ids=input_ids, attention_mask=attention_mask, images=images)
        return logits

def make_lime_predict_fn(model, input_ids, attention_mask, transform_val, device):
    """
    Creates prediction function for LIME image explainer
    Converts LIME's numpy arrays to model-compatible tensor format
    Maintains fixed text context while perturbing image inputs for local explanations
    """
    def pred_fn(images_np):
        model.eval()
        tensors = []
        # Convert each numpy array in batch to preprocessed tensor
        for im in images_np:
            # Convert numpy array to PIL Image for consistent preprocessing
            pil = Image.fromarray(im.astype('uint8'), 'RGB')
            # Apply same transforms as validation pipeline
            x = transform_val(pil).unsqueeze(0)
            tensors.append(x)
        # Combine all processed images into single batch tensor
        batch = torch.cat(tensors, dim=0).to(device)
        # Prepare repeated text inputs for batch
        b = batch.shape[0]
        ids = input_ids.unsqueeze(0).repeat(b,1).to(device)
        mask = attention_mask.unsqueeze(0).repeat(b,1).to(device)
        # Get model predictions and convert to probabilities
        with torch.no_grad():
            logits, *_ = model(ids, mask, batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    return pred_fn

def explain_image_with_lime_and_gradcam(model, tokenizer,
                                        input_ids, attention_mask, image_tensor,
                                        transform_val,
                                        device='cuda',
                                        lime_samples=50,
                                        top_label=None,
                                        save_path=None):
    """
    Main explanation function combining Grad-CAM and LIME for multimodal image analysis
    Generates both gradient-based and perturbation-based explanations for model decisions
    Returns visualization figure and explanation metadata for further analysis
    """
    model.eval()
    # Prepare image tensor on correct device
    image = image_tensor.clone().detach().to(device)
    
    # Reverse normalization to reconstruct original image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    im_np = image.cpu().numpy().transpose(1,2,0)
    im_01 = (im_np * std + mean).clip(0,1)

    # --- Grad-CAM Explanation ---
    # Select target layer from ViT encoder for gradient computation
    target_layer = model.image_encoder.encoder.layer[-1].output
    # Create wrapper model for Grad-CAM with fixed text inputs
    wrapper = MultiModalForCAM(model, input_ids.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device))
    wrapper.to(device)
    # Initialize Grad-CAM explainer with ViT reshape transformation
    cam = GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape_transform)
    
    # Get model prediction for target category selection
    with torch.no_grad():
        inp = image.unsqueeze(0).to(device)
        logits, *_ = model(input_ids=input_ids.unsqueeze(0).to(device),
                           attention_mask=attention_mask.unsqueeze(0).to(device),
                           images=inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_label = int(probs.argmax(axis=1)[0])
    
    # Determine target category for explanation
    target_category = pred_label if top_label is None else int(top_label)
    # Generate Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=inp, targets=None)[0]
    # Overlay heatmap on original image
    cam_image = show_cam_on_image(im_01, grayscale_cam, use_rgb=True)

    # --- LIME Explanation ---
    # Initialize LIME image explainer
    explainer = lime_image.LimeImageExplainer()
    # Create prediction function with fixed text context
    predict_fn = make_lime_predict_fn(model, input_ids, attention_mask, transform_val, device)
    # Convert image to uint8 format for LIME processing
    im_for_lime = (im_01 * 255).astype('uint8')
    # Generate LIME explanation using superpixel segmentation
    explanation = explainer.explain_instance(
        im_for_lime,
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=lime_samples,          
        segmentation_fn=lambda x: slic(x, n_segments=25, compactness=5)
    )
    # Extract LIME mask for target category
    temp, mask = explanation.get_image_and_mask(target_category,
                                                positive_only=False,
                                                num_features=10,
                                                hide_rest=False)
    lime_overlay = temp

    # --- Visualization ---
    import matplotlib.pyplot as plt
    # Create three-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    
    # Original image panel
    axes[0].imshow((im_01.clip(0,1)))
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Grad-CAM explanation panel
    axes[1].imshow(cam_image)
    axes[1].set_title(f'Grad-CAM (pred={pred_label})')
    axes[1].axis('off')

    # LIME explanation panel
    axes[2].imshow(lime_overlay)
    axes[2].set_title('LIME overlay')
    axes[2].axis('off')

    plt.tight_layout()
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        
    return fig, {'pred_prob': probs[0].tolist(), 'pred_label': pred_label, 'grayscale_cam': grayscale_cam, 'lime_mask': mask}