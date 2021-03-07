# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:15:12 2020

@author: TW
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# from torchvision.models import inception_v3
from torchvision import transforms 
from dataset import BasicDataset
import torch
import argparse
from PIL import Image
import xml.etree.ElementTree as ET
import sys
import openslide
import time
###xmax,ymax=65500
Image.MAX_IMAGE_PIXELS = 10000000000

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=160, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='test', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint', type=str, default='123/20.pth', help='checkpoint file')
opt = parser.parse_args()
print(opt)

checkpoint_path='model/'+ opt.checkpoint
print(checkpoint_path)

img_path=opt.dataroot+'/image'
image_list=os.listdir(img_path)
n=len(image_list)
image_list.sort(key=lambda x:int(x.split('.')[0][-3:]))

label_path=opt.dataroot+'/label'
label_list=os.listdir(label_path)
m=len(label_list)
label_list.sort(key=lambda x:int(x.split('.')[0][-3:]))

def crop_tumor(root,slide,savepath):
    region={}
    k=0
    for a in root.iter('Annotations'):
        for b in a.iter('Annotation'):
            str_b = ET.tostring(b, encoding='unicode', method='xml')
            title = str_b.split('>')[0].split(' ')
            # print(title)
            for c in b.iter('Coordinates'):
                A=[]
                for i in range(len(c)):
                    B=[]
                    X,Y=get_coordinates(c[i])
                    # print(X,Y)
                    B.append(X)
                    B.append(Y)
                    A.append(B)
                l=np.array(A)
                x1=np.min(l[:,0])
                x2=np.max(l[:,0])
                y1=np.min(l[:,1])
                y2=np.max(l[:,1])
                # print(x1,y1,x2,y2)
                x0=int(x1)
                y0=int(y1)
                img=slide.read_region((x0,y0),0,(int(x2-x0)+1,int(y2-y0)+1)).convert('RGB')
                img=np.array(img)[:,:,::-1]
                cv2.imwrite(savepath+'/{}.jpg'.format(k), img)
                for j in range(len(c)):
                    A[j][0]=A[j][0]-x0
                    A[j][1]=A[j][1]-y0
                region[k]=A
                k=k+1
    return region

def get_coordinates(xml):
    str_xml = ET.tostring(xml, encoding='unicode', method='xml')
    Coordinates=str_xml.split(' ')
    # print(Coordinates)
    X=Coordinates[2][3:-1]
    Y=Coordinates[3][3:-1]            
    return  float(X), float(Y)

def crop_img(img_path,img_list,crop_path):
    cropsize=300
    step=300
    k0=0
    crop_path0=crop_path+'/0'
    crop_path1=crop_path+'/1'
    if not os.path.exists(crop_path0):
        os.makedirs(crop_path0)
    if not os.path.exists(crop_path1):
        os.makedirs(crop_path1)
    for k in range(len(img_list)):
        img = Image.open(img_path + '/' + img_list[k])
        x_Num=int((img.size[0]-cropsize)/step)+1
        y_Num=int((img.size[1]-cropsize)/step)+1

        for j in range(y_Num):
            for i in range(x_Num):
                a = step * i   # 图片距离左边的大小
                b = step * j # 图片距离上边的大小
                c = a+cropsize  # 图片距离左边的大小 + 图片自身宽度
                d = b+cropsize  # 图片距离上边的大小 + 图片自身高度
                img_cropped = img.crop((a, b, c, d))  # (left, upper, right, lower)
                average = np.asarray(img_cropped).mean()
                p=255-average
                if (p<30):
                    img_cropped.save(crop_path0+'/{}.jpg'.format(k0))
                    k0=k0+1
                else:
                    img_cropped.save(crop_path1+'/{}.jpg'.format(k0))
                    k0=k0+1                
    return x_Num, y_Num


time0=time.time()
for i in range(m):
    svs_path= img_path+'/'+ image_list[i]
    xml_path= label_path+'/'+ label_list[i]
    slide = openslide.open_slide(svs_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    path='test/result/'+image_list[i].split('.')[0]
    path1=path+'/original'
    if not os.path.exists(path1):
        os.makedirs(path1)
    print(path)
    region=crop_tumor(root,slide,path1)
    ####crop region
    img_list=os.listdir(path1)
    img_list.sort(key=lambda x: int(x.split('.')[0]))
    print(img_list)
    path2=path+'/cropped'  
    if not os.path.exists(path2):
        os.makedirs(path2)
    x_Num, y_Num = crop_img(path1,img_list,path2)
    mat = np.zeros(shape=(y_Num,x_Num))
    print('crop finish!')

    transform = transforms.Compose(
    [
    #   transforms.Resize(opt.size, Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    )


    dataset = BasicDataset(imgs_dir=path2+'/1',transform=transform)
    # print(len(dataset))
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True, shuffle=False)
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
    print('test finish')
    # print(result)
    # print(probability)

    filelist = [file.split('.')[0] for file in os.listdir(path2+'/1')]
    filelist.sort(key=lambda x:int(x))
    # print(filelist)
    for j in range(len(result)):
        number=int(filelist[j])
        a= number // x_Num
        b= number % x_Num
        if(result[j]==0):
            mat[a][b]= 0.5
        elif(result[j]==1):
            mat[a][b]= 1

    print('draw heatmap0')
    path3=opt.dataroot+'/heatmap/'+image_list[i].split('.')[0]
    if not os.path.exists(path3):
        os.makedirs(path3)

    plt.close()
    plt.imshow(mat, cmap='rainbow')
    plt.xticks([])
    plt.yticks([])
    cbar=plt.colorbar()
    cbar.set_ticks([0,0.5,1])
    cbar.set_ticklabels(['background','normal','tumor'])
    plt.savefig(path3+'/0.jpg')


    cropsize=300
    step=300
    blank_image = np.zeros(((y_Num-1) * step+cropsize,(x_Num-1)*step+cropsize,3), np.uint8)
    # print(blank_image.shape)
    list0=os.listdir(path2+'/0')
    for j in range(y_Num):
        for i in range(x_Num):
            if '{}.jpg'.format(j*x_Num+i) in list0:
                from_image = cv2.imread(path2+'/0' + '/{}.jpg'.format(j*x_Num+i))
            else:
                from_image = cv2.imread(path2+'/1' + '/{}.jpg'.format(j*x_Num+i))
            
            blank_image[j * step:j * step+cropsize,i * step:i * step+cropsize,:]=from_image

    # cv2.imwrite('11.jpg', blank_image)
    # print(mat)
    mask=mat
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap1= cv2.resize(heatmap, (heatmap.shape[1]*300, heatmap.shape[0]*300))
    # print(heatmap1.shape)
    # cv2.imwrite('110.jpg', np.uint8(heatmap1*255))

    img1 = heatmap1 + np.float32(blank_image/255)
    img1 -= np.max(np.min(img1), 0)
    img1 /= np.max(img1)
    img1 *= 255.

    print('draw heatmap1')
    cv2.imwrite(path3+'/1.jpg', np.uint8(img1))

time1=time.time()   
print('{}m {}s'.format((time1-time0)//60,(time1-time0)%60))  
