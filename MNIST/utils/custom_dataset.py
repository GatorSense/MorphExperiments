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
        
class ThreesAndKMNIST(Dataset):
    def __init__(self, kmnist, threes):
        self.kmnist = kmnist
        self.threes = threes
    
    def __len__(self):
        return len(self.kmnist) + len(self.threes)
    
    def __getitem__(self, index):
        if index < len(self.threes):
            image, _ = self.threes[index]
            return image, 1
        else:
            return self.kmnist[index - len(self.threes)], 0
        
def generate_hitmiss_morphed_filters(train_subset_3, rand_index, kernel, show_plots=False):
    filter_indices = list(rand_index)
    dilated_filters = []
    eroded_filters = []
    for idx in filter_indices:
        img, label = train_subset_3[idx]  # assuming each item returns (image, label)
        if show_plots:
            plt.imshow(img[0], cmap='gray')
            plt.show()

        img = img.unsqueeze(0)  # Add batch dimension if needed, shape becomes [1, C, H, W]
        dilated_img = kornia.morphology.dilation(img, kernel)
        dilated_filters.append((dilated_img.squeeze(0), label))  # Remove batch dim if needed
        eroded_img = kornia.morphology.erosion(img, kernel)
        eroded_filters.append((eroded_img.squeeze(0), label))

        if show_plots:
            plt.imshow(dilated_img[0][0], cmap='gray')
            plt.show()
            plt.imshow(eroded_img[0][0], cmap='gray')
            plt.show()

    return [dilated_filters, eroded_filters]