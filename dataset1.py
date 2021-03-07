import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
import time
from torchvision import transforms 

class TrainDataset(Dataset):
    def __init__(self, imgs_dir, list1, list2, r1, r2, is_train, transform=None):
        self.imgs_dir = imgs_dir
        self.transform1 = transform
        self.normal_path=imgs_dir+'/normal'
        self.tumor_path=imgs_dir+'/tumor'
        self.l1=len(r1)
        self.l2=len(r2)
        self.img_list=[]
        self.label_list = torch.LongTensor(self.l1*[0]+self.l2*[1])
        for i in range(len(r1)):
            self.img_list.append(self.normal_path+'/'+list1[r1[i]])
        for i in range(len(r2)):
            self.img_list.append(self.tumor_path+'/'+list2[r2[i]])
        self.class_to_idx={'normal':'0','tumor':'1'}
        self.transform2 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04)
            ]
        )
        self.is_train=is_train
    def __len__(self):

        return self.l1 + self.l2

    def __getitem__(self, i):
        img_file=self.img_list[i]
        img_label=self.label_list[i]
        img = Image.open(img_file)
        if (self.is_train):        
            if img_label.item()==1:
                number=random.randint(1,4)
                # print(number)
                if number==1:
                    img=img
                elif number==2:
                    img=img.rotate(90)
                elif number==3:
                    img=img.rotate(180)
                elif number==4:
                    img=img.rotate(270)
                img = self.transform2(img)
        if self.transform1 is not None:
            img = self.transform1(img)
        return (img,img_label)
 


class twodataset():
    def __init__(self, imgs_dir,split_ratio,transform=None):
        time0=time.time()
        normal_path=imgs_dir+'/normal'
        tumor_path=imgs_dir+'/tumor'
        list1= os.listdir(normal_path)
        time1=time.time()
        print('read normal list {} time:{}'.format(len(list1),time1-time0))
        list2= os.listdir(tumor_path)
        time2=time.time()
        print('read tumor list {}  time:{}'.format(len(list2),time2-time1))
        list1.sort(key=lambda x:int(x.split('.')[0]))
        list2.sort(key=lambda x:int(x.split('.')[0]))
        a1=np.random.permutation(len(list1))
        a2=np.random.permutation(len(list2))
        b1=int(len(list1)*split_ratio)
        b2=int(len(list2)*split_ratio)
        s1=a1[:b1]
        s2=a2[:b2]
        s3=a1[b1:]
        s4=a2[b2:]
        self.train_dataset=TrainDataset(imgs_dir, list1, list2,r1=s1,r2=s2, is_train=1,transform=transform)
        self.val_dataset=TrainDataset(imgs_dir, list1, list2,r1=s3,r2=s4,is_train=0,transform=transform)

    def makedataset(self):
        
        return (self.train_dataset,self.val_dataset)        
        
# aaa=twodataset(imgs_dir='datasets/2011',split_ratio=0.7)
# train_dataset,val_dataset=aaa.makedataset()

# print(len(train_dataset))
# print(len(val_dataset))

# from tqdm import tqdm
# import time
# for i in tqdm(range(100)):
#     time.sleep(0.01)