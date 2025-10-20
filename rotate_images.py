import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import math

# === CONFIGURATION ===
labels_csv = 'data/leaves/labels.csv'
image_dir = 'data/leaves/images'
output_dir = 'data/leaves/rotated_images'
os.makedirs(output_dir, exist_ok=True)

# === LOAD LABELS CSV ===
df = pd.read_csv(labels_csv)

# === HELPER FUNCTIONS ===

# Parse range like '1001-1059'
def parse_range(r):
    start, end = map(int, str(r).split('-'))
    return list(range(start, end + 1))

# Rotate and optionally pad
def rotate_and_pad(img, angle):
    if angle % 90 == 0:
        return img.rotate(angle, expand=True, fillcolor='white')
    
    # Convert to radians
    angle_rad = math.radians(angle)

    # Original size
    w, h = img.size

    # Compute size of bounding box after rotation
    cos_theta = abs(math.cos(angle_rad))
    sin_theta = abs(math.sin(angle_rad))
    new_w = int(w * cos_theta + h * sin_theta)
    new_h = int(w * sin_theta + h * cos_theta)

    # Create new white background image
    new_img = Image.new("RGB", (new_w, new_h), (255, 255, 255))

    # Paste original image at center
    new_img.paste(img, ((new_w - w) // 2, (new_h - h) // 2))

    # Rotate the new image
    return new_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor='white')

# === PROCESS EACH LABEL GROUP ===
for _, row in tqdm(df.iterrows(), total=len(df), desc='Rotating images'):
    rotate_deg = int(row['rotate'])
    image_range = parse_range(row['filename'])

    for img_id in image_range:
        img_path = os.path.join(image_dir, f'{img_id}.jpg')
        out_path = os.path.join(output_dir, f'{img_id}.jpg')

        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            rotated_img = rotate_and_pad(img, rotate_deg)
            rotated_img.save(out_path)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

print(f"\n✅ All rotated images saved to: {output_dir}")
