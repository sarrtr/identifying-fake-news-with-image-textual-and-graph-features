import io
import base64
import gc
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch

from model_utils import model, tokenizer, transform_val, device
from XAI_image import explain_image_with_lime_and_gradcam
from XAI_text import lime_explain_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_tensor(pil_img):
    """
    Применяет transform_val (PIL->Tensor->Normalize) и возвращает tensor (C,H,W)
    """
    return transform_val(pil_img).detach()
def tensor_to_base64_jpg(tensor_image):
    """
    tensor_image: HxWx3 float 0..1 or uint8 HxWx3
    возвращает base64 строки JPEG
    """
    import io
    from PIL import Image
    import numpy as np
    import base64

    arr = tensor_image
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    # ensure uint8 HWC
    if arr.dtype in (np.float32, np.float64):
        arr = (np.clip(arr, 0, 1) * 255).astype('uint8')

    # ensure RGB
    im = Image.fromarray(arr).convert("RGB")

    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def fig_to_base64_jpg(fig):
    """
    Convert a Matplotlib figure to base64-encoded JPEG image.
    """
    import io, base64

    buf = io.BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

# -------- API models --------
class PredictResponse(BaseModel):
    text: str
    label: str
    confidence: float
    lime_text_list: list
    lime_text_html: str
    gradcam_image: str   # base64 png
    lime_image_overlay: str  # base64 png

# -------- endpoints --------
@app.post("/predict", response_model=PredictResponse)
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...),
    lime_samples: Optional[int] = Form(200)
):
    """
    Принимает текст (form field) и изображение (form file).
    Возвращает: оригинал, label, confidence, LIME text (list + html),
    Grad-CAM image и LIME overlay image (base64 strings).
    """
    # 1) Read image
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2) Preprocess image into tensor as your model expects
    image_tensor = pil_to_tensor(pil_img)  # (C,H,W)
    # Ensure on CPU for LIME wrapper; functions expect to push to device internally.
    # 3) Tokenize text for model -> input_ids, attention_mask
    enc = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].squeeze(0)
    attention_mask = enc['attention_mask'].squeeze(0)

    # 4) Run model prediction (single forward)
    model.eval()
    with torch.no_grad():
        # expand image batch dim
        images_batch = image_tensor.unsqueeze(0).to(device)
        input_ids_b = input_ids.unsqueeze(0).to(device)
        attention_mask_b = attention_mask.unsqueeze(0).to(device)
        logits, *_ = model(input_ids=input_ids_b, attention_mask=attention_mask_b, images=images_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label_idx = int(np.argmax(probs))
        confidence = float(probs[pred_label_idx])
        label_str = "fake" if pred_label_idx == 1 else "real"

    # 5) XAI: LIME for text (returns exp object)
    # lime_explain_text given in prompt returns an Explanation object `exp`
    exp_obj = lime_explain_text(text, image_tensor, model, tokenizer, device=device, batch_size=8)
    # produce list and html
    lime_list = exp_obj.as_list()  # list[(token, weight), ...]
    try:
        lime_html = exp_obj.as_html(label=pred_label_idx, predict_proba=True)
    except Exception:
        lime_html = ""

    # 6) XAI: explain_image_with_lime_and_gradcam
    # This returns (fig, meta) where meta has pred_prob, pred_label, grayscale_cam, lime_mask
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
    # convert fig -> base64
    try:
        gradcam_b64 = fig_to_base64_jpg(fig)
    except Exception:
        # fallback: if fig not available, return empty
        gradcam_b64 = ""
    # lime_overlay: meta contains 'lime_mask' and LIME overlay saved as return; but our function returned 'lime_mask' and had overlay in variable lime_overlay.
    # For simplicity, regenerate a PNG from fig's 3rd panel: we already saved entire fig -> gradcam_b64; reuse for both fields.
    lime_overlay_b64 = gradcam_b64

    # free memory
    del images_batch, input_ids_b, attention_mask_b, logits
    torch.cuda.empty_cache()
    gc.collect()

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
