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
parser.add_argument('--dataroot', type=str, default='test1', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint', type=str, default='217/10.pth', help='checkpoint file')
opt = parser.parse_args()
print(opt)

checkpoint_path='model/'+ opt.checkpoint
print(checkpoint_path)

data_path=opt.dataroot+'/data'

slide_path=data_path+'/slide'
image_list=os.listdir(slide_path)
n=len(image_list)

label_path=data_path+'/label'
label_list=os.listdir(label_path)
m=len(label_list)

def crop_tumor(root,slide,path):
    region={}
    k=0
    for a in root.iter('Annotations'):
        for b in a.iter('Annotation'):
            str_b = ET.tostring(b, encoding='unicode', method='xml')
            title = str_b.split('>')[0].split(' ')
            group=title[-2].split('"')[-2]
            if group=='Tumor':
                group='0'
            elif group=='Exclusion':
                group='2'
            elif group=='None':
                group='9'    
            else:
                group=group[-1]    

            for c in b.iter('Coordinates'):
                A=[]
                for i in range(len(c)):
                    B=[]
                    X,Y=get_coordinates(c[i])
                    # print(X,Y)
                    B.append(X)
                    B.append(Y)
                    A.append(B)
                region[k]=A
                k=k+1
                l=np.array(A)
                x1=np.min(l[:,0])
                x2=np.max(l[:,0])
                y1=np.min(l[:,1])
                y2=np.max(l[:,1])
                # print(x1,y1,x2,y2)
                x0=int(x1)
                y0=int(y1)
                w=int(x2-x0)+1
                h=int(y2-y0)+1
                img=slide.read_region((x0,y0),0,(w,h)).convert('RGB')
                img=img.resize((img.size[0]*2,img.size[1]*2))
                img.save(path+'/slide1.jpg')

    for i in region:
        for j in range (len(region[i])):
            region[i][j][0]=region[i][j][0]-x0
            region[i][j][1]=region[i][j][1]-y0

    return region, img

def get_coordinates(xml):
    str_xml = ET.tostring(xml, encoding='unicode', method='xml')
    Coordinates=str_xml.split(' ')
    # print(Coordinates)
    X=Coordinates[2][3:-1]
    Y=Coordinates[3][3:-1]            
    return  float(X), float(Y)

def crop_img(img,crop_path):
    k0=0
    cropsize=300
    step=300
    crop_path0=crop_path+'/0'
    crop_path1=crop_path+'/1'
    if not os.path.exists(crop_path0):
        os.makedirs(crop_path0)
    if not os.path.exists(crop_path1):
        os.makedirs(crop_path1)
    print(img.size)
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
            if (p<50):
                img_cropped.save(crop_path0+'/{}.jpg'.format(k0))
                k0=k0+1
            else:
                img_cropped.save(crop_path1+'/{}.jpg'.format(k0))
                k0=k0+1                
    return x_Num, y_Num


time0=time.time()
for i in range(m):
    xml_path= label_path+'/'+ label_list[i]
    svs_name=label_list[i].split('.')[0]  
    print(svs_name) 
    svs_path= slide_path+'/'+ svs_name+'.ndpi'
    slide = openslide.open_slide(svs_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result_path = opt.dataroot+'/results'
    path=result_path + '/detail/' + label_list[i].split('.')[0]
    if not os.path.exists(path):
        os.makedirs(path)
    region,slide_image= crop_tumor(root,slide,path) 
   

    ####crop region
    time2=time.time()  
    path3=path+'/cropped'  
    if not os.path.exists(path3):
        os.makedirs(path3)
    x_Num, y_Num = crop_img(slide_image,path3)
    time3=time.time()
    print('crop time {}m {}s'.format((time3-time2)//60,(time3-time2)%60))    
    print(x_Num, y_Num)
    # mat = np.zeros(shape=(y_Num,x_Num))

    ### model inference
    transform = transforms.Compose([
    #   transforms.Resize(opt.size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = BasicDataset(imgs_dir=path3+'/1',transform=transform)
    # print(len(dataset))
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=False, shuffle=False)
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
    print('test finish!')

    filelist = [file.split('.')[0] for file in os.listdir(path3+'/1')]
    filelist.sort(key=lambda x:int(x))
    # print(filelist)

    mat1=np.zeros(shape=(y_Num,x_Num))
    mat2=np.zeros(shape=(y_Num,x_Num))
    ####
    for j in range(len(result)):
        number=int(filelist[j])
        a= number // x_Num
        b= number % x_Num
        if(result[j]==0):
            mat1[a][b]= 0.5
        elif(result[j]==1):
            mat1[a][b]= 1

    ### tumor probability heatmap
    excel_path=result_path+'/probability'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    csv_path=excel_path+'/'+label_list[i].split('.')[0]+'.csv'
    fconv = open(csv_path, 'w')
    fconv.write('Xcorr, Ycorr, Probs\n')
    fconv.close()
    for j in range(len(result)):
        number=int(filelist[j])
        a= number // x_Num
        b= number % x_Num
        mat2[a][b]= probability[j][1]
        fconv = open(csv_path, 'a')
        fconv.write('{},{},{}\n'.format(a,b,mat2[a][b]))
        fconv.close()

    path4=result_path+'/heatmap/'+label_list[i].split('.')[0]
    if not os.path.exists(path4):
        os.makedirs(path4)
    heatmap1 = cv2.applyColorMap(np.uint8(255 * mat1), cv2.COLORMAP_JET)
    heatmap2 = cv2.applyColorMap(np.uint8(255 * mat2), cv2.COLORMAP_JET)
    # heatmap1= cv2.resize(heatmap1, (mask_size[0], mask_size[1]))    
    # heatmap2= cv2.resize(heatmap2, (mask_size[0], mask_size[1]))   
    cv2.imwrite(path4+'/1.jpg',heatmap1)
    cv2.imwrite(path4+'/2.jpg',heatmap2)
    print('heatmap finish!')


time1=time.time()   
print('{}m {}s'.format((time1-time0)//60,(time1-time0)%60))  
