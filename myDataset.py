import os
import torch
import torch.utils.data as data
from PIL import Image
from multiprocessing.dummy import Pool

def default_loader(path):
    return Image.open(path).convert('RGB')

class myImageFolder(data.Dataset):
    def __init__(self, label_file, transform = None, target_transform=None, loader=default_loader):        
        with open(label_file,'r') as f:
            fh = f.read().splitlines()[:100]
        imgs=[]
        pool_ = Pool(100)
        for line in fh:
            pool_.apply_async(self.load_one_image, (line, imgs))
        pool_.close()
        pool_.join()
        print('load {} images done'.format(len(imgs)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def load_one_image(self, line, imgs):
        cls = line.split()
        fn = cls.pop(0)
        if os.path.isfile(fn):
            imgs.append((fn, tuple([float(v) for v in cls])))