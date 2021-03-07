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
# from visdom import Visdom
import torchvision
from tqdm import tqdm
from dataset1 import twodataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='numbers of epoch')
parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/304', help='root directory of the dataset')
parser.add_argument('--size', type=int, default=300, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--train_last', action='store_true', help='continue train')
parser.add_argument('--checkpoint', type=str, default='217/10.pth', help='checkpoint file')
opt = parser.parse_args()
print(opt)

transform = transforms.Compose(
    [
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.5),
    # transforms.ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

traindataset=twodataset(imgs_dir=opt.dataroot,split_ratio=0.9,transform=transform)
train_dataset,val_dataset=traindataset.makedataset()
# print(train_dataset.label_list)
print('train_dataset:{} val_dataset:{}'.format(len(train_dataset),len(val_dataset)))


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=opt.batchSize, num_workers=opt.n_cpu, pin_memory=True, shuffle=True)

def fit(epoch, model, data_loader, optimizer,loss_f, phase='training'):
    if phase == 'training':
        model.train()
        print('train')
    if phase == 'validation':
        model.eval()
        print('val')
    running_loss = 0.0
    running_corrects = 0
    t0=time.time()
    for i, (images, labels) in enumerate(data_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
            
        # 梯度归零
        optimizer.zero_grad()

        # 计算损失
        loss = loss_f(outputs,labels)
        if phase=='training':
            loss.backward()
            optimizer.step()
                  
        # 计算损失和
        running_loss += float(loss)
            
        # 统计预测正确的图片数
        running_corrects += torch.sum(predicted==labels.data).item()
        
        if( i%10000 == 0 and i!=0):
            t1=time.time()
            print('Batch:{}/{} Time:{}m {}s'.format(i,len(data_loader),(t1-t0)//60,(t1-t0)%60))
            t0=t1
            
    # print(outputs.size())
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = 100 * float(running_corrects)/ len(data_loader.dataset)
    print("{} Loss: {:.4f} Acc: {:.4f}%".format(phase, epoch_loss, epoch_acc))
 
    return epoch_loss, epoch_acc 

#### 2 class
if opt.train_last:
    checkpoint_path='model/'+ opt.checkpoint
    print('load checkpoint from: '+ checkpoint_path)
    model = torch.load(checkpoint_path)
else:
    model=inception_v3(pretrained=True)
    model.aux_logits=False
    model.fc = torch.nn.Linear(2048, 2)
    # model = torchvision.models.resnet50(pretrained=True) 
    # model.fc = torch.nn.Linear(2048, 2)

if opt.cuda:
    model.cuda()

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss=[]
train_accuracy=[]
val_loss=[]
val_accuracy=[]

epoch_n = opt.epoch
# bestmodel=model
best_acc=0.0

checkpoint_savepath='model/'+ opt.dataroot.split('/')[1]
isExist=os.path.exists(checkpoint_savepath)
if not isExist:
    os.makedirs(checkpoint_savepath)
excel_path='metric/'+ opt.dataroot.split('/')[1]
if not os.path.exists(excel_path):
    os.makedirs(excel_path)
fconv = open(os.path.join(excel_path,'convergence.csv'), 'w')
fconv.write('epoch,training accuracy, training loss, validation accurcay, validation loss\n')
fconv.close()

# viz=Visdom()
if opt.train_last:
    start_epoch=int(opt.checkpoint.split('/')[1].split('.')[0])
else:
    start_epoch=0
finish_epoch=start_epoch+opt.epoch
print(start_epoch,finish_epoch)

for epoch in tqdm(range(start_epoch,finish_epoch)):
    time0=time.time()
    print('')
    print("Epoch {}/{}".format(epoch+1,finish_epoch))
    print("-"*10)
    epoch_loss, epoch_accuracy=fit(epoch,model,train_loader,optimizer,loss_f, phase='training')
    time1=time.time()   
    print('{}m {}s'.format((time1-time0)//60,(time1-time0)%60))    
    val_epoch_loss, val_epoch_accuracy=fit(epoch,model,val_loader,optimizer,loss_f, phase='validation')
    time2=time.time()     
    print('{}m {}s'.format((time2-time1)//60,(time2-time1)%60))      
    # loss_name1 = 'training loss'
    # acc_name1 = 'training accuracy'
    # loss_name2 = 'validation loss'
    # acc_name2 = 'validation accuracy'   
    # if (epoch==0):
    #     win1=viz.line(X=np.array([epoch]),Y=np.array([epoch_loss]),opts={'xlabel':'epoch','ylabel':loss_name1,'title':loss_name1})
    #     win2=viz.line(X=np.array([epoch]),Y=np.array([epoch_accuracy]),opts={'xlabel':'epoch','ylabel':acc_name1,'title':acc_name1})
    #     win3=viz.line(X=np.array([epoch]),Y=np.array([val_epoch_loss]),opts={'xlabel':'epoch','ylabel':loss_name2,'title':loss_name2})
    #     win4=viz.line(X=np.array([epoch]),Y=np.array([val_epoch_accuracy]),opts={'xlabel':'epoch','ylabel':acc_name2,'title':acc_name2})
    # else:
    #     viz.line(X=np.array([epoch]),Y=np.array([epoch_loss]),win=win1,update='append')
    #     viz.line(X=np.array([epoch]),Y=np.array([epoch_accuracy]),win=win2,update='append')
    #     viz.line(X=np.array([epoch]),Y=np.array([val_epoch_loss]),win=win3,update='append')
    #     viz.line(X=np.array([epoch]),Y=np.array([val_epoch_accuracy]),win=win4,update='append')

    acc=0.7*epoch_accuracy+0.3*val_epoch_accuracy
    print('overall_acc: {:.4f}'.format(acc))
    if (acc >= best_acc):
        best_epoch = epoch + 1
        best_acc = acc
    train_loss.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    fconv = open(os.path.join(excel_path, 'convergence.csv'), 'a')
    fconv.write('{},{},{},{},{}\n'.format(epoch+1,epoch_accuracy,epoch_loss,val_epoch_accuracy,val_epoch_loss))
    fconv.close()
    torch.save(model, checkpoint_savepath+'/{}.pth'.format(epoch+1))
    # torch.save(model.state_dict(),opt.checkpoint+'{}.pth'.format(epoch+1))

print('best_epoch: {}'.format(best_epoch))

fig_path='figure/'+ opt.dataroot.split('/')[1]
isExist=os.path.exists(fig_path)
if not isExist:
    os.makedirs(fig_path)

plt.plot(range(1,opt.epoch+1),train_loss,label='training loss')
plt.plot(range(1,opt.epoch+1),val_loss,label='validation loss')
plt.legend()
plt.savefig(fig_path+'/'+'loss.jpg')

plt.close()
plt.plot(range(1,opt.epoch+1),train_accuracy,label='training accuracy')
plt.plot(range(1,opt.epoch+1),val_accuracy,label='validation accuracy')
plt.legend()
plt.savefig(fig_path+'/'+'accuracy.jpg')


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

# for i, (images, labels) in enumerate(train_loader):
#     print()