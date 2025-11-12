import os
import torch
import pandas as pd
from tqdm import tqdm
from predict_single import predict_single

def evaluate_dataset(dataset_path, device=None, limit=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text_file = os.path.join(dataset_path, "text", "text.txt")
    image_dir = os.path.join(dataset_path, "images")

    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    data = pd.read_csv(text_file, sep='\t')
    if limit:
        data = data.sample(n=min(limit, len(data)), random_state=42).reset_index(drop=True)
        print(f"Evaluating on subset: {len(data)} samples")

    results = []
    progress = tqdm(data.iterrows(), total=len(data), desc="Evaluating MiRAGe")

    for _, row in progress:
        image_path = os.path.join(image_dir, f"{row['id']}.jpg")
        text = str(row['clean_title'])

        if not os.path.exists(image_path):
            tqdm.write(f"[WARN] Missing image: {image_path}")
            continue

        try:
            out = predict_single(image_path, text, device)
            results.append({
                "id": row["id"],
                "label_true": int(row["2_way_label"]),
                "prob_image": out["prob_image"],
                "prob_text": out["prob_text"],
                "prob_combined": out["prob_combined"],
                "pred_label": out["label"],
            })
        except Exception as e:
            tqdm.write(f"[ERROR] {row['id']}: {e}")
            continue

    df = pd.DataFrame(results)
    out_path = os.path.join(dataset_path, "mirage_eval_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

    if not df.empty:
        acc = (df["label_true"] == (df["pred_label"] == "fake").astype(int)).mean()
        print(f"MiRAGe accuracy: {acc:.4f} ({len(df)} samples evaluated)")

    return df


if __name__ == "__main__":
    dataset_path = "/repo/project_deepfake/project/fakeddit_dataset"
    evaluate_dataset(dataset_path, limit=100)
