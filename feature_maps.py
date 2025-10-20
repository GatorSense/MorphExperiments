import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import OrderedDict
from PIL import Image

def read_metadata_expand_ranges(csv_path):
    df = pd.read_csv(csv_path)
    filename_to_meta = {}
    for _, row in df.iterrows():
        start, end = map(int, str(row['filename']).split('-'))
        for i in range(start, end + 1):
            # fname = f"{i}.jpg"
            fname = f"{i}.jpg"
            filename_to_meta[fname] = {
                'label': int(row['label']),
                'scientific_name': row['Scientific Name'],
                'common_name': row['Common Name(s)'],
                'url': row['URL'],
            }
    all_filenames = sorted(filename_to_meta.keys())
    return all_filenames, filename_to_meta
    
class LeafDataset(Dataset):
    def __init__(self, filenames, filename_to_meta, image_dir, transform=None,
                 mode='train'):

        self.image_dir = image_dir
        self.transform = transform
        self.filename_to_meta = filename_to_meta
        self.mode = mode

        if mode == 'train':
            self.filenames = [f for f in filenames]
        else:
            self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        meta  = self.filename_to_meta[fname]
        img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)

        original_label = int(meta['label'])
        target = original_label - 1

        # Return both for analysis; training will use `target`
        return img, torch.tensor(target, dtype=torch.long), original_label, fname

# --- 1) Load model + weights ---
from pytorchcifar.models import resnet
model = resnet.ResNet50()

checkpoint = torch.load('pytorchcifar/checkpoint/ckpt.pth', map_location='cpu')
state = checkpoint.get('net', checkpoint)  # handle either {'net': ...} or plain state_dict

# If checkpoint was saved under DataParallel, strip 'module.' prefixes.
if any(k.startswith('module.') for k in state.keys()):
    new_state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())
    state = new_state

model.load_state_dict(state)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# --- 2) Use the model's backbone (everything except the final FC) ---
# This returns (N, 2048, 1, 1) which we'll flatten to (N, 2048).
backbone = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
backbone.eval()

transform=transforms.Compose([ 
    transforms.Resize((128, 128)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)), 
    transforms.RandomRotation((0, 360)) 
])

all_filenames, filename_to_meta = read_metadata_expand_ranges('/blue/azare/samgallic/Research/MorphExperiments/data/leaves/labels.csv')
traindataset = LeafDataset(all_filenames, filename_to_meta, '/blue/azare/samgallic/Research/MorphExperiments/data/leaves/images', transform=transform, mode='train')
trainloader = DataLoader(traindataset, batch_size=16, shuffle=True, num_workers=4)
testdataset = LeafDataset(all_filenames, filename_to_meta, '/blue/azare/samgallic/Research/MorphExperiments/data/leaves/images', transform=transform, mode='test')
testloader = DataLoader(testdataset, batch_size=64, shuffle=True, num_workers=4)
loader = trainloader

# --- 4) Extract + save features ---
out_dir = "/blue/azare/samgallic/Research/MorphExperiments/data/leaves/trained_resnet"
os.makedirs(out_dir, exist_ok=True)

records = []  # for manifest

@torch.inference_mode()
def dump_features():
    for imgs, targets, orig_labels, fnames in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = backbone(imgs)          # (B, 2048, 1, 1)
        feats = torch.flatten(feats, 1) # (B, 2048)
        feats = feats.cpu()

        for vec, fname, orig_lab, targ in zip(feats, fnames, orig_labels, targets):
            # Save per image
            save_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}.pt")
            torch.save(vec, save_path)

            records.append({
                "filename": fname,
                "label_original": int(orig_lab),
                "label_zero_indexed": int(targ.item()),
                "feature_path": save_path
            })

dump_features()

# --- 5) Write a manifest CSV for convenience ---
manifest_path = os.path.join(out_dir, "manifest.csv")
pd.DataFrame.from_records(records).to_csv(manifest_path, index=False)

print(f"✓ Saved features to: {out_dir}")
print(f"✓ Manifest: {manifest_path}")
