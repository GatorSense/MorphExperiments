import random
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia

class BlackAndThrees(Dataset):
    def __init__(self, black_imgs, threes):
        self.black_imgs = black_imgs
        self.threes = threes
    
    def __len__(self):
        return len(self.black_imgs) + len(self.threes)
    
    def __getitem__(self, index):
        if index < len(self.threes):
            image, _ = self.threes[index]
            return image, 1
        else:
            return self.black_imgs[index - len(self.threes)], 0
        
class FilterOutThrees(Dataset):
    def __init__(self, black_imgs, threes, filter_3s):
        self.black_imgs = black_imgs
        self.threes = threes
        self.filter_3s = filter_3s
    
    def __len__(self):
        return len(self.black_imgs) + len(self.threes) + len(self.filter_3s)
    
    def __getitem__(self, index):
        if index < len(self.threes):
            image, _ = self.threes[index]
            return image, 1
        elif index >= len(self.threes) and index < (len(self.threes) + len(self.filter_3s)):
            image, _ = self.filter_3s[index - len(self.threes)]
            return image, 2
        else:
            return self.black_imgs[index - len(self.threes) - len(self.filter_3s)], 0
        
class ThreesAndNotThree(Dataset):
    def __init__(self, not_three, threes):
        self.not_three = not_three
        self.threes = threes
    
    def __len__(self):
        return len(self.not_three) + len(self.threes)
    
    def __getitem__(self, index):
        if index < len(self.threes):
            image, _ = self.threes[index]
            return image, 1
        else:
            image, _ = self.not_three[index - len(self.threes)]
            return image, 0
        
class TripletThreesAndNotThree(Dataset):
    def __init__(self, not_three, threes):
        assert len(threes) > 1 and len(not_three) > 1, \
            "Both classes need at least two samples to form triplets."
        self.not_three = not_three
        self.threes = threes
        self._len = max(len(threes), len(not_three))

    def __len__(self):
        return self._len * 2           # one pass with 3-as-anchor, one with not-3

    def _sample_positive(self, dataset, exclude_idx):
        """Randomly pick a different element from the same dataset."""
        pos_idx = random.randrange(len(dataset) - 1)
        if pos_idx >= exclude_idx:
            pos_idx += 1               # skip the anchors position
        img, _ = dataset[pos_idx]
        return img

    def __getitem__(self, index):
        use_three_as_anchor = index % 2 == 0          # alternate for balance
        if use_three_as_anchor:
            anchor_idx = index // 2 % len(self.threes)
            anchor_img, _ = self.threes[anchor_idx]
            positive_img         = self._sample_positive(self.threes, anchor_idx)
            negative_img, _      = self.not_three[random.randrange(len(self.not_three))]
        else:
            anchor_idx = index // 2 % len(self.not_three)
            anchor_img, _ = self.not_three[anchor_idx]
            positive_img         = self._sample_positive(self.not_three, anchor_idx)
            negative_img, _      = self.threes[random.randrange(len(self.threes))]

        return anchor_img, positive_img, negative_img

class TripletThreesAndNotThreeWithLabel(TripletThreesAndNotThree):
    def __getitem__(self, index):
        anchor, positive, negative = super().__getitem__(index)

        y = torch.tensor(1 if index % 2 == 0 else 0, dtype=torch.long)
        return anchor, positive, negative, y

def generate_hitmiss_morphed_filters(train_subset_3, rand_index, kernel, show_plots=False):
    filter_indices = list(rand_index)
    dilated_filters = []
    eroded_filters = []
    for idx in filter_indices:
        img, label = train_subset_3[idx] # Assuming each item returns (image, label)
        if show_plots:
            plt.imshow(img[0], cmap='gray')
            plt.show()

        img = img.unsqueeze(0) # Add batch dimension if needed, shape becomes [1, C, H, W]
        dilated_img = kornia.morphology.dilation(img, kernel)
        dilated_filters.append((dilated_img.squeeze(0), label)) # Remove batch dim if needed
        eroded_img = kornia.morphology.erosion(img, kernel)
        eroded_filters.append((eroded_img.squeeze(0), label))

        if show_plots:
            plt.imshow(dilated_img[0][0], cmap='gray')
            plt.show()
            plt.imshow(eroded_img[0][0], cmap='gray')
            plt.show()

    return [dilated_filters, eroded_filters]