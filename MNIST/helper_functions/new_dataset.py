from torch.utils.data import Dataset

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