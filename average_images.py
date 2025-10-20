import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# === CONFIGURATION ===
labels_csv = 'data/leaves/labels.csv'
image_dir = 'data/leaves/rotated_images'  # Change to your actual image folder
output_dir = 'average_rotated_images'
image_size = (224, 224)  # Resize to consistent dimensions

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# === LOAD LABELS ===
df = pd.read_csv(labels_csv)

# Helper: parse filename ranges like '1001-1059' to list of ints
def parse_range(s):
    start, end = map(int, s.split('-'))
    return list(range(start, end + 1))

# === PROCESS EACH LABEL ===
for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing labels'):
    label_name = row['Scientific Name']
    filename_range = row['filename']
    image_ids = parse_range(str(filename_range))

    avg_img = None
    count = 0

    for img_id in image_ids:
        file_path = os.path.join(image_dir, f'{img_id}.jpg')
        if not os.path.exists(file_path):
            continue

        img = Image.open(file_path).convert('RGB').resize(image_size)
        img_array = np.array(img, dtype=np.float32)

        if avg_img is None:
            avg_img = img_array
        else:
            avg_img += img_array

        count += 1

    if count == 0:
        print(f"⚠️ No images found for label: {label_name}")
        continue

    # Compute average
    avg_img /= count
    avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)
    avg_image_pil = Image.fromarray(avg_img)

    # Save with safe filename
    safe_name = label_name.replace(' ', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f'{safe_name}.jpg')
    avg_image_pil.save(output_path)

print(f"\n✅ Saved average images for all labels in: {output_dir}")
