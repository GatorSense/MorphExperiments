from torch.utils.data import Dataset

class BinaryLeafDataset(Dataset):
    def __init__(self, filenames, filename_to_meta, image_dir, transform=None,
                 mode='train'):
        """
        mode='train': only samples with label in {POS_LABEL} ∪ NEG_LABELS
        mode='test' : all samples; targets mapped to binary on-the-fly
        """
        self.image_dir = image_dir
        self.transform = transform
        self.filename_to_meta = filename_to_meta
        self.mode = mode

        if filenames[0].endswith('.pt'):
            self.type = 'fms'
        else:
            self.type = 'images'

        if mode == 'train':
            self.filenames = [f for f in filenames
                              if filename_to_meta[f]['label'] == POS_LABEL
                              or filename_to_meta[f]['label'] in NEG_LABELS]
        else:
            self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if self.type == 'images':
            fname = self.filenames[idx]
            meta  = self.filename_to_meta[fname]
            img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
            if self.transform:
                img = self.transform(img)

            original_label = int(meta['label'])
            # Binary target mapping
            target = 1 if original_label == POS_LABEL else 0

            # Return both for analysis; training will use `target`
            return img, torch.tensor(target, dtype=torch.long), original_label, fname

        else:
            fname = self.filenames[idx]
            meta  = self.filename_to_meta[fname]

            path = os.path.join(self.image_dir, fname)
            # safe load for PyTorch 2.6
            img = torch.load(path, weights_only=False)

            # Ensure float tensor
            if not torch.is_tensor(img):
                img = torch.tensor(img)
            img = img.float()

            # --- STANDARDIZE LAYOUT TO [C, H, W] ---
            if img.ndim == 3:
                C, H, W = img.shape  # could be wrong order
                # If looks like [H, W, C] (channels last)
                if img.shape[-1] in (1, 3, 2048) and img.shape[0] == img.shape[1]:
                    img = img.permute(2, 0, 1)  # HWC -> CHW
                # If looks like [H, C, W]
                elif (img.shape[1] in (1, 3, 2048)) and (img.shape[0] == img.shape[2]):
                    img = img.permute(1, 0, 2)  # HCW -> CHW

            original_label = int(meta['label'])
            target = 1 if original_label == POS_LABEL else 0
            return img, torch.tensor(target, dtype=torch.long), original_label, fname

