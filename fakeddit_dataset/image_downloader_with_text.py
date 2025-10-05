import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request

FILE_NAME = 'data/multimodal_test_public.tsv'
MAX_SAMPLES = 100

df = pd.read_csv(FILE_NAME, sep="\t")
df = df.fillna('')

pbar = tqdm(total=len(df))
download_counter = 0

if not os.path.exists("images"):
    os.makedirs("images")

if not os.path.exists("text"):
    os.makedirs("text")
    
with open("text/text.txt", "w", encoding="utf-8") as text_file:
    text_file.write("id\tclean_title\t2_way_label\n")
    
    for index, row in df.iterrows():
        if download_counter >= MAX_SAMPLES:
            break

        if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan" and row["id"] != "" and row["clean_title"] != "":
            try:
                image_url = row["image_url"]
                urllib.request.urlretrieve(image_url, "images/" + row["id"] + ".jpg")
                text_file.write(f"{row['id']}\t{row['clean_title']}\t{row['2_way_label']}\n")
                download_counter += 1
            except Exception as e:
                print(f"Error while downloading {image_url}: {e}")
        
        pbar.update(1)
    
pbar.close()
print("done")