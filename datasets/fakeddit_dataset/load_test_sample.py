import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = 'project_deepfake/project/datasets/fakeddit_dataset' 

# Load the dataset file
txt_file = os.path.join(dataset_path, 'text', 'text.txt')
df = pd.read_csv(txt_file, sep='\t')  

train_idx, val_idx = train_test_split(
    range(len(df)),
    test_size=0.2,
    random_state=42,
    stratify=df['2_way_label'] 
)

val_df = df.iloc[val_idx]

val_sampled_df = val_df.sample(n=1500, random_state=42)

new_txt_dir = os.path.join(dataset_path, 'text_sampled')
os.makedirs(new_txt_dir, exist_ok=True)
val_sampled_df.to_csv(os.path.join(new_txt_dir, 'text.txt'), sep='\t', index=False)

src_image_path = 'project_deepfake/project/fakeddit_dataset' 
image_dir = os.path.join(src_image_path, 'images')
new_image_dir = os.path.join(dataset_path, 'images_sampled')
os.makedirs(new_image_dir, exist_ok=True)

for img_name in val_sampled_df['id']:
    src_path = os.path.join(image_dir, img_name+'.jpg')
    dst_path = os.path.join(new_image_dir, img_name+'.jpg')
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
