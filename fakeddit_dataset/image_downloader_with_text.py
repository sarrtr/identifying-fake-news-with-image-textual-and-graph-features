import pandas as pd
import os
from tqdm import tqdm
import urllib.request
import time
import random
import re

FILE_NAME = 'data/multimodal_test_public.tsv'
MAX_SAMPLES = 100

df = pd.read_csv(FILE_NAME, sep="\t")
df = df.fillna('')

if not os.path.exists("images"):
    os.makedirs("images")
if not os.path.exists("text"):
    os.makedirs("text")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

def imgur_to_direct(url):
    if "imgur.com" in url and not re.search(r'i\.imgur\.com', url):
        match = re.search(r'imgur\.com/([A-Za-z0-9]+)', url)
        if match:
            return f"https://i.imgur.com/{match.group(1)}.jpg"
    return url


def download_with_retry(url, filename, max_retries=2, backoff_factor=2, min_delay=2, max_delay=5):
    url = imgur_to_direct(url)
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req) as response:
                with open(filename, 'wb') as f:
                    f.write(response.read())
            print(f"Downloaded: {filename}")
            return True

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = backoff_factor * attempt + random.uniform(min_delay, max_delay)
                print(f"429 Too many requests. Sleeping {wait_time:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"HTTP {e.code}: {url}")
                break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(random.uniform(min_delay, max_delay))

    print(f"Failed after {max_retries} attempts: {url}")
    return False


pbar = tqdm(total=len(df))
download_counter = 0

with open("text/text.txt", "w", encoding="utf-8") as text_file:
    text_file.write("id\tclean_title\t2_way_label\n")

    for _, row in df.iterrows():
        if download_counter >= MAX_SAMPLES:
            break

        image_url = str(row.get("image_url", "")).strip()
        if (
            row.get("hasImage", False)
            and image_url
            and image_url.lower() != "nan"
            and row.get("id", "")
            and row.get("clean_title", "")
        ):
            out_path = os.path.join("images", f"{row['id']}.jpg")

            success = download_with_retry(image_url, out_path)
            if success:
                text_file.write(f"{row['id']}\t{row['clean_title']}\t{row['2_way_label']}\n")
                download_counter += 1

        pbar.update(1)

pbar.close()
print("done")
print(f"{download_counter} samples downloaded successfully.")
