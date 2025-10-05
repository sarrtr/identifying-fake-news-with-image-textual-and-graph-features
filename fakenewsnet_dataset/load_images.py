import requests
import os
import pandas as pd
from tqdm import tqdm

def download_images(df, img_col="images", out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)
    df['installed_img'] = ''
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_list = row[img_col].split(',')
        for j, url in enumerate(img_list):
            try:
                r = requests.get(url, timeout=0.5)
                if r.status_code == 200:
                    filename = f"{i}_{j}.jpg"
                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                    img_list[j] = filename
                else:
                    img_list[j] = None
            except Exception as e:
                print(f"Ошибка при загрузке {url}: {e}")
                img_list[j] = None
        print(str([x for x in img_list if x is not None]))
        df.at[i, 'installed_img'] = ','.join([x for x in img_list if x is not None])

    return df

df = pd.read_csv("fakenewsnet_cleaned.csv")
df = download_images(df, img_col="images", out_dir="images")
df.to_csv("fakenewsnet_cleaned.csv", index=False)
