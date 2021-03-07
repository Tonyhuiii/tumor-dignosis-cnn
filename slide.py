# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:15:12 2020

@author: TW
"""

# import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from torchvision.models import inception_v3
from torchvision import transforms 
from dataset import BasicDataset
import torch
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=160, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='2.jpg', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint', type=str, default='217/10.pth', help='checkpoint file')
opt = parser.parse_args()
print(opt)


img=Image.open(opt.dataroot)
cropsize=300
step=300
x_Num=int((img.size[0]-cropsize)/step)+1
y_Num=int((img.size[1]-cropsize)/step)+1
print(x_Num,y_Num)


path='slide/'+opt.dataroot.split('.')[0]
path0=path+'/0'
path1=path+'/1'
print(path)
if not os.path.exists(path0):
    os.makedirs(path0)
if not os.path.exists(path1):
    os.makedirs(path1)

k0=0
for i in range(y_Num):
    for j in range(x_Num):
        a = step * j   # 图片距离左边的大小
        b = step * i # 图片距离上边的大小
        c = a+cropsize  # 图片距离左边的大小 + 图片自身宽度
        d = b+cropsize  # 图片距离上边的大小 + 图片自身高度
        img_cropped = img.crop((a, b, c, d))
        average = np.asarray(img_cropped).mean()
        p=255-average
        # cropped = img[b:d, a:c]  # 裁剪坐标为[y0:y1, x0:x1]
        # p=255-cropped.mean()
        if (p<30):
            img_cropped.save(path0+'/{}.jpg'.format(k0))
            # cv2.imwrite(path0+'/{}.jpg'.format(k0),cropped)
            k0=k0+1
        else:
            img_cropped.save(path1+'/{}.jpg'.format(k0))
            # cv2.imwrite(path1+'/{}.jpg'.format(k0),cropped)
            k0=k0+1

print('crop finish!')

transform = transforms.Compose(
 [
#   transforms.Resize(opt.size, Image.BICUBIC),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ]
)

dataset = BasicDataset(imgs_dir=path1,transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True, shuffle=False)

checkpoint_path='model/'+ opt.checkpoint
model = torch.load(checkpoint_path)
if opt.cuda:
    model.cuda()

def test(model,data_loader):
    model.eval()
    result=[]
    probability=[]
    for i, images in enumerate(data_loader):
        images = images.cuda()
        with torch.no_grad():
            outputs = model(images)
            matrix= torch.nn.functional.softmax(outputs,dim=1)
            _, predicted = torch.max(outputs.data, 1)
            result.extend(predicted.cpu().tolist())
            probability.extend(matrix.cpu().tolist())
            
    return result, probability

result, probability = test(model,test_loader)
print('inference finish!')


filelist = [file.split('.')[0] for file in os.listdir(path1)]
filelist.sort(key=lambda x:int(x))
# print(filelist)

mat= np.zeros(shape=(y_Num,x_Num))
for i in range(len(result)):
    number=int(filelist[i])
    a= number // x_Num
    b= number % x_Num
    if(result[i]==0):
        mat[a][b]= 0.5
    elif(result[i]==1):
        mat[a][b]= 1

path2='heatmap'
if not os.path.exists(path2):
    os.makedirs(path2)

plt.close()
plt.imshow(mat, cmap='rainbow')
plt.xticks([])
plt.yticks([])
cbar=plt.colorbar()
cbar.set_ticks([0,0.5,1])
cbar.set_ticklabels(['background','normal','tumor'])
plt.savefig(path2+'/' + opt.dataroot.split('.')[0] + '-0.jpg')


# blank_image = np.zeros(((y_Num-1) * step+cropsize,(x_Num-1)*step+cropsize,3), np.uint8)
# list0=os.listdir(path0)
# for j in range(y_Num):
#     for i in range(x_Num):
#         if '{}.jpg'.format(j*x_Num+i) in list0:
#             from_image = cv2.imread(path0 + '/{}.jpg'.format(j*x_Num+i))
#         else:
#             from_image = cv2.imread(path1 + '/{}.jpg'.format(j*x_Num+i))
        
#         blank_image[j * step:j * step+cropsize,i * step:i * step+cropsize,:]=from_image

# cv2.imwrite('11.jpg', blank_image)

# mask=mat
# heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
# heatmap = np.float32(heatmap) / 255

# heatmap1= cv2.resize(heatmap, (heatmap.shape[1]*300, heatmap.shape[0]*300))
# # print(heatmap1.shape)
# # cv2.imwrite('110.jpg', np.uint8(heatmap1*255))

# img1 = heatmap1 + np.float32(blank_image/255)
# img1 -= np.max(np.min(img1), 0)
# img1 /= np.max(img1)
# img1 *= 255.

# cv2.imwrite(path2+'/{}-1.jpg'.format(opt.dataroot.split('.')[0]), np.uint8(img1))


# print(slide.level_dimensions)
# print(slide.level_downsamples)
# print(slide.level_count)

# #得到原图的缩略图（206X400）
# simg = slide.get_thumbnail((800,800))

# #simg=img.convert('L')
# #显示缩略图
# plt.imshow(simg,cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.show()
