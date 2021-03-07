from os.path import splitext
from os import listdir
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        # print(self.ids)
        self.ids.sort(key=lambda x:int(x))
        # print(self.ids)

        # logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):

        return len(self.ids)


    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = self.imgs_dir + '/' + idx + '.jpg'
        # print(img_file)
        img = Image.open(img_file)

        if self.transform is not None:
            img = self.transform(img)

        return img

