from lime.lime_text import LimeTextExplainer
import torch
import numpy as np
import gc

explainer = LimeTextExplainer(class_names=['real', 'fake'])

def lime_explain_text(sample_text, image_tensor, model, tokenizer, device='cuda', batch_size=8):
    """
    Explain text contribution in a multimodal model (text + fixed image).
    GPU-safe version with batching and cache clearing.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Prepare image
    image_tensor = image_tensor.unsqueeze(0).to(device)

    @torch.no_grad()
    def wrapped_predict(text_list):
        """
        LIME will call this many times.
        We run it in mini-batches to avoid CUDA OOM.
        """
        probs_all = []

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]

            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(device, non_blocking=True)

            # Repeat image for each text
            images = image_tensor.repeat(len(batch_texts), 1, 1, 1)

            # Forward
            outputs = model(input_ids, attention_mask, images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)

            # Free CUDA memory after each batch
            del input_ids, attention_mask, images, outputs, logits
            torch.cuda.empty_cache()
            gc.collect()

        return np.concatenate(probs_all, axis=0)

    # LIME explanation (this still runs on CPU)
    exp = explainer.explain_instance(
        sample_text,
        wrapped_predict,
        num_features=10,
        labels=(0, 1)
    )

    return exp
