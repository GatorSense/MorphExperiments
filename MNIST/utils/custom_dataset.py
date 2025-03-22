from torch.utils.data import Dataset
import torch

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