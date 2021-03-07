from torchvision.models import inception_v3
from torchvision import transforms 
from torchvision.datasets import ImageFolder
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from PIL import Image
# import torchvision

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=5, help='numbers of epoch')
parser.add_argument('--batchSize', type=int, default=20, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='tumor/datasets/1127', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
# parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
# parser.add_argument('--checkpoint', type=str, default='model/1120/', help='checkpoint file')
opt = parser.parse_args()
print(opt)


transform = transforms.Compose(
 [
#   transforms.Resize((opt.size,opt.size)),
#   transforms.Resize(int(opt.size*1.1), Image.BICUBIC),
#   transforms.RandomCrop(300),
  transforms.RandomHorizontalFlip(0.5),
  transforms.RandomVerticalFlip(0.5),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 ]
)

dataset = ImageFolder(root=opt.dataroot,transform=transform)
print(dataset.class_to_idx)
print(len(dataset))
a=int(len(dataset)*0.7)
b=len(dataset)-a
train_dataset, val_dataset= torch.utils.data.random_split(dataset, [a, b])
# val_dataset = ImageFolder(root=val_root,transform=transform)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True, shuffle=True)

model=inception_v3(pretrained=False)
model.aux_logits=False
model.fc = torch.nn.Linear(2048, 2)
model.cuda()

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]

epoch_n = opt.epoch
# bestmodel=model
best_acc= 0.0

checkpoint_path='model/'+ opt.dataroot.split('/')[1]
isExist=os.path.exists(checkpoint_path)
if not isExist:
    os.makedirs(checkpoint_path)

# viz=Visdom()

for epoch in range(0,opt.epoch):
    since=time.time()
    print('')
    print("Epoch {}/{}".format(epoch+1,opt.epoch))
    print("-"*10)

    ####train
    running_loss = 0.0
    running_corrects = 0
    model.train()
    print('train')
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
            
        # 梯度归零
        optimizer.zero_grad()
        # 计算损失
        loss = loss_f(outputs , labels)
        loss.backward()
        optimizer.step()       
        # 计算损失和
        running_loss += float(loss)  
        # 统计预测正确的图片数
        running_corrects += torch.sum(predicted==labels.data).item()  
    # print(outputs.size())
    train_epoch_loss = running_loss / len(train_loader.dataset)
    train_epoch_acc = 100 * float(running_corrects)/ len(train_loader.dataset)
    print("{} Loss: {:.4f} Acc: {:.4f}%".format('train', train_epoch_loss, train_epoch_acc))

    ####val
    running_loss = 0.0
    running_corrects = 0
    model.eval()
    print('val')
    for i, (images, labels) in enumerate(val_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = loss_f(outputs , labels)
        running_loss += float(loss)
        running_corrects += torch.sum(predicted==labels.data).item()

    val_epoch_loss = running_loss / len(val_loader.dataset)
    val_epoch_acc = 100 * float(running_corrects)/ len(val_loader.dataset)

    print("{} Loss: {:.4f} Acc: {:.4f}%".format('val', val_epoch_loss, val_epoch_acc))

    time_elapsed=time.time()-since
    print('{}m {}s'.format( time_elapsed//60 , time_elapsed%60 ))

    acc = 0.7 * train_epoch_acc + 0.3 * val_epoch_acc
    print('overall_acc: {:.4f}'.format(acc))
    if (acc >= best_acc):
        best_epoch = epoch + 1
        best_acc = acc
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_acc)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_acc)
    torch.save(model, checkpoint_path+'/{}.pth'.format(epoch+1))

print('best_epoch: {}'.format(best_epoch))


plt.plot(range(1,opt.epoch+1),train_loss,label='training loss')
plt.plot(range(1,opt.epoch+1),val_loss,label='validation loss')
plt.legend()
plt.savefig('loss.jpg')

plt.close()
plt.plot(range(1,opt.epoch+1),train_accuracy,label='training accuracy')
plt.plot(range(1,opt.epoch+1),val_accuracy,label='validation accuracy')
plt.legend()
plt.savefig('accuracy.jpg')


# def fit(epoch, model, data_loader, optimizer,loss_f, phase='training'):
#     if phase == 'training':
#         model.train()
#         print('train')
#     if phase == 'validation':
#         model.eval()
#         print('val')
#     running_loss = 0.0
#     running_corrects = 0
        
#     for i, (images, labels) in enumerate(data_loader):
#         images = images.cuda()
#         labels = labels.cuda()
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
            
#         # 梯度归零
#         optimizer.zero_grad()

#         # 计算损失
#         loss = loss_f(outputs , labels)
#         if phase=='training':
#             loss.backward()
#             optimizer.step()
                    
#         # 计算损失和
#         running_loss += float(loss)
            
#         # 统计预测正确的图片数
#         running_corrects += torch.sum(predicted==labels.data).item()
        
#     # print(outputs.size())
#     epoch_loss = running_loss / len(data_loader.dataset)
#     epoch_acc = 100 * float(running_corrects)/ len(data_loader.dataset)
#     print("{} Loss: {:.4f} Acc: {:.4f}%".format(phase, epoch_loss, epoch_acc))

#     return epoch_loss, epoch_acc 



# def imshow(image):
#     image=image.numpy().transpose((1,2,0))
#     mean=np.array([0.485, 0.456, 0.406]) 
#     std=np.array([0.229, 0.224, 0.225])
#     image=std*image+mean
#     image=np.clip(image,0,1)
#     plt.imshow(image)
#     plt.axis('off')
#     plt.savefig('1.jpg')

# image=train_dataset[30][0]
# imshow(image)
# print(image.shape)    
