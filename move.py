import os
import shutil
import time 
from PIL import Image

path='datasets/304/1'
file_list=os.listdir(path)
file_list.sort(key=lambda x:int(x.split('.')[0]))
m=len(file_list)
print(m)
path1='datasets/304/normal'

#重命名
time0=time.time()
for i in range(m):
    name=int(file_list[i].split('.')[0])
    file_name1=path+'/{}.jpg'.format(name)
    file_name2=path+'/{}.jpg'.format(name+3000000)
    shutil.move(file_name1, file_name2)
time1=time.time()
print(time1-time0)

#移动文件
time0=time.time()
file_list=os.listdir(path)
file_list.sort(key=lambda x:int(x.split('.')[0]))
for i in range(m):
    file_name=path+'/'+file_list[i]
    shutil.move(file_name, path1)
time1=time.time()
print(time1-time0)



# # path='datasets/205/normal'
# path='datasets/train/crop/normal'
# file_list=os.listdir(path)
# file_list.sort(key=lambda x:int(x.split('.')[0]))
# b=len(file_list)
# # a=Image.open(path+'/'+file_list[10])
# # a.show()

# #复制单个文件
# shutil.copy("C:\\a\\1.txt","C:\\b")
# #复制并重命名新文件
# shutil.copy("C:\\a\\2.txt","C:\\b\\121.txt")
# #复制整个目录(备份)
# shutil.copytree("C:\\a","C:\\b\\new_a")
 
# #删除文件
# os.unlink("C:\\b\\1.txt")
# os.unlink("C:\\b\\121.txt")
# #删除空文件夹
# try:
#     os.rmdir("C:\\b\\new_a")
# except Exception as ex:
#     print("错误信息："+str(ex))#提示：错误信息，目录不是空的
# #删除文件夹及内容
# shutil.rmtree("C:\\b\\new_a")
 
# #移动文件
# shutil.move("C:\\a\\1.txt","C:\\b")
# #移动文件夹
# shutil.move("C:\\a\\c","C:\\b")
 
# #重命名文件
# shutil.move("C:\\a\\2.txt","C:\\a\\new2.txt")
# #重命名文件夹
# shutil.move("C:\\a\\d","C:\\a\\new_d")