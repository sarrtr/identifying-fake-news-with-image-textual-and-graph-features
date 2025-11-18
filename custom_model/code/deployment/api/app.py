import io
import gc
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
import time

from model_utils import model, tokenizer, transform_val, device
from XAI_image import explain_image_with_lime_and_gradcam
from XAI_text import lime_explain_text

app = FastAPI()

# Configure CORS to allow cross-origin requests from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_tensor(pil_img):
    """
    Convert PIL image to preprocessed tensor using validation transforms
    Returns tensor in (C,H,W) format ready for model input
    """
    return transform_val(pil_img).detach()

def tensor_to_base64_jpg(tensor_image):
    """
    Convert tensor image to base64-encoded JPEG string
    Handles both float (0-1) and uint8 tensors, ensures RGB format
    """
    import io
    from PIL import Image
    import numpy as np
    import base64

    arr = tensor_image
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # Normalize float tensors to uint8 range
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype('uint8')

    # Convert to PIL Image and ensure RGB format
    im = Image.fromarray(arr).convert("RGB")

    # Encode to JPEG and convert to base64
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def fig_to_base64_jpg(fig):
    """
    Convert matplotlib figure to base64-encoded JPEG image
    Uses high DPI and tight bounding box for quality output
    """
    import io, base64

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

# Response model for prediction endpoint
class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    lime_text_list: list
    lime_text_html: str
    gradcam_image: str
    lime_image_overlay: str

@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...),
    lime_samples: Optional[int] = Form(200)
):
    """
    Main prediction endpoint for multimodal fake news detection
    Accepts text content and image file, returns prediction with XAI explanations
    Includes LIME text analysis, Grad-CAM and LIME image explanations
    """
    start = time.time()
    
    # Read and process uploaded image
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Convert PIL image to preprocessed tensor
    image_tensor = pil_to_tensor(pil_img)

    # Tokenize input text for model processing
    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].squeeze(0)
    attention_mask = enc['attention_mask'].squeeze(0)

    # Run model inference to get prediction
    model.eval()
    with torch.no_grad():
        # Prepare batch inputs for model
        images_batch = image_tensor.unsqueeze(0).to(device)
        input_ids_b = input_ids.unsqueeze(0).to(device)
        attention_mask_b = attention_mask.unsqueeze(0).to(device)
        
        # Get model predictions and convert to probabilities
        logits, *_ = model(input_ids=input_ids_b, attention_mask=attention_mask_b, images=images_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_idx = int(np.argmax(probs))
        confidence = float(probs[pred_label_idx])
        label_str = "fake" if pred_label_idx == 1 else "real"
    print('Model evaluation completed in', time.time() - start, 'seconds')

    # Generate LIME explanations for text
    exp_obj = lime_explain_text(text, image_tensor, model, tokenizer, device=device, batch_size=8)
    print('LIME text explanation completed in', time.time() - start, 'seconds')
    
    # Extract token weights and HTML visualization from LIME
    lime_list = exp_obj.as_list()
    lime_html = exp_obj.as_html(labels=(pred_label_idx,))

    # Generate image explanations using LIME and Grad-CAM
    fig, meta = explain_image_with_lime_and_gradcam(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_tensor=image_tensor,
        transform_val=transform_val,
        device=device,
        lime_samples=lime_samples,
        top_label=pred_label_idx,
        save_path=None
    )
    print('Image explanation completed in', time.time() - start, 'seconds')
    
    # Convert explanation figures to base64 for web display
    try:
        gradcam_b64 = fig_to_base64_jpg(fig)
    except Exception:
        gradcam_b64 = ""
    lime_overlay_b64 = gradcam_b64

    # Clean up memory to prevent GPU memory leaks
    del images_batch, input_ids_b, attention_mask_b, logits
    torch.cuda.empty_cache()
    gc.collect()

    # Construct and return prediction response
    response = PredictResponse(
        text=text,
        label=label_str,
        confidence=confidence,
        lime_text_list=lime_list,
        lime_text_html=lime_html,
        gradcam_image=gradcam_b64,
        lime_image_overlay=lime_overlay_b64
    )
    return JSONResponse(content=response.model_dump())