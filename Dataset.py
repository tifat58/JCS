import torch
import cv2
import os.path as osp
import torch.utils.data
import numpy as np 
import os

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def scaleRadius(img, scale) :
    x=img[int(img.shape[0]/2),:,:].sum(1)
#     print(x)
    r=(x>x.mean()/10).sum()/2
    s=scale * 1.0 / r
#     print(r, s)
    return cv2.resize(img,(0,0), fx=s, fy=s), r, s

def scaleRadius_mask(img, scale, r, s) :
    x=img[int(img.shape[0]/2),:,:].sum(1)
#     print(x)
#     r=(x>x.mean()/10).sum()/2
#     s=scale * 1.0 / r
#     print(r, s)
    img = cv2.resize(img,(0,0), fx=s, fy=s)
    img[img > 0] = 255
    return img
    
class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, dataset, transform=None):
        self.transform = transform
        self.img_list = list()
        self.msk_list = list()
        dataset_name = 'IDRID'
        with open(osp.join(data_dir, dataset + '.txt'), 'r') as lines:
            for line in lines:
                if dataset_name == 'IDRID':
                    line_arr = line.split(',')
                else:
                    line_arr = line.split()
                
                
                self.img_list.append(osp.join(data_dir, line_arr[0].strip()))
                self.msk_list.append(osp.join(data_dir, line_arr[1].strip('\n')))

    def __len__(self):
        return len(self.img_list)
    
    

    def __getitem__(self, idx):
#         if os.path.isfile(self.msk_list[idx]) == False:
#             print("No file")
        scale = 500
        image1 = cv2.imread(self.img_list[idx])
        image, r, s = scaleRadius(image1, scale)
        image=cv2.addWeighted (image , 4 , cv2.GaussianBlur( image , ( 0 , 0 ) , scale /30) , -4 , 128)

        
        label1 = cv2.imread(self.msk_list[idx])
        label = scaleRadius_mask(label1, scale, r, s)
        label = label[:, :, 2]
#         print(np.amax(label), np.amin(label))
#         exit()
        if self.transform:
            [image, label] = self.transform(image, label)
#         print(image.shape, label.shape)
        return image, label

    def get_img_info(self, idx):
        image = cv2.imread(self.img_list[idx])
        return {"height": image.shape[0], "width": image.shape[1]}
