# from torchvision.models import inception_v3
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch
import numpy as np
import os
import argparse
from PIL import Image
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=200, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/117', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint', type=str, default='117/10.pth', help='checkpoint file')
opt = parser.parse_args()
print(opt)

time0=time.time()  
checkpoint_path='model/'+ opt.checkpoint
print(checkpoint_path)

transform = transforms.Compose(
 [
#   transforms.Resize((opt.size,opt.size), Image.BICUBIC),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ]
)

dataset = ImageFolder(root=opt.dataroot,transform=transform)
print(dataset.class_to_idx)
print(len(dataset))
test_dataset, val_dataset= torch.utils.data.random_split(dataset, [len(dataset), 0])


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True, shuffle=False)

# print(len(test_dataset))
# print(test_dataset.class_to_idx)
# print(test_dataset[0][0].shape)



model = torch.load(checkpoint_path)
# model=inception_v3(pretrained=False)
# model.aux_logits=False
# model.fc = torch.nn.Linear(2048, 2)
# model.load_state_dict(torch.load(checkpoint_path))
# if opt.cuda:
model.cuda()

def test(model,data_loader):
    model.eval()
    running_corrects = 0
    # result=[]
    # label=[]
    # probability=[]
    for i, (images, labels) in enumerate(data_loader):
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
            matrix= torch.nn.functional.softmax(outputs,dim=1)
            _, predicted = torch.max(outputs.data, 1)
            # result.extend(predicted.cpu().tolist())
            # label.extend(labels.cpu().tolist())
            # probability.extend(matrix.cpu().tolist())
            running_corrects += torch.sum(predicted==labels.data).item()
        print(i,len(data_loader))
            
    # print(result)
    epoch_acc = 100 * float(running_corrects) / len(data_loader.dataset)
    print("Acc: {:.4f}%".format(epoch_acc))
    
#    print(outputs.size())
    return  epoch_acc

test_accuracy = test(model,test_loader)

print('test_accuracy:{}'.format(test_accuracy))

time1=time.time()   
print('{}m {}s'.format((time1-time0)//60,(time1-time0)%60))   

'''
img_dir = 'breast cancer/PA/3he/'
test_dataset = BasicDataset(img_dir,transform=transform)
print(test_dataset.ids)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
'''