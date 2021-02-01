#coding=utf-8
import jittor as jt 
from jittor import dataset
import numpy as np 
import json
import os
import glob
from PIL import Image
from utils.box_ops import BBox 
from .transforms import train_transforms,val_transforms
    
def read_annotations(filename,filter_empty=False,classnames=None):
    annotations = json.load(open(filename))
    if classnames is None:
        classnames = annotations['types']
    imgs = annotations['imgs']
    test_imgs = []
    train_imgs = []
    other_imgs = []
    all_imgs = []
    for img in imgs.values():
        if filter_empty and len([o for o in img['objects'] if o['category'] in classnames])==0:
            continue
        path = img['path']
        if 'train' in path:
            train_imgs.append(img)
        elif 'test' in path:
            test_imgs.append(img)
        else:
            other_imgs.append(img)
        all_imgs.append(img)
    return train_imgs,test_imgs,all_imgs,classnames


class TrainDataset(dataset.Dataset):
    def __init__(self,data_dir,annos,classnames,transforms=None,batch_size=1,shuffle=False,num_workers=0):
        super(TrainDataset,self).__init__(batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.total_len = len(annos)
        self.annos = annos
        self.data_dir = data_dir
        self.classnames = classnames
        self.transforms = transforms
    
    def __getitem__(self,index):
        anno = self.annos[index]
        img_path = os.path.join(self.data_dir,anno['path'])
        objects = [o for o in anno['objects'] if o['category'] in self.classnames]
        xyxy = [[o['bbox']['xmin'],o['bbox']['ymin'],o['bbox']['xmax'],o['bbox']['ymax']] for o in objects]
        labels = [self.classnames.index(o['category'])+1 for o in objects]
        img = Image.open(img_path)
        ori_img_size = img.size 
        boxes = BBox(xyxy,img_size=img.size)
        labels = np.array(labels,dtype=np.int32)
        if self.transforms is not None:
            img,boxes = self.transforms(img,boxes)
        return img.astype(np.float32),boxes.bbox,labels,ori_img_size,anno['id']
    
    def collate_batch(self,batch):
        imgs = []
        boxes = []
        labels = []
        img_sizes = []
        img_ids = []
        for img,box,label,img_size,ID in batch:
            imgs.append(img)
            boxes.append(box.astype(np.float32))
            labels.append(label)
            img_sizes.append(img_size)
            img_ids.append(ID)
        imgs = np.stack(imgs,axis=0)
        return imgs,boxes,labels,img_sizes,img_ids

class TestDataset(dataset.Dataset):
    def __init__(self,img_dir,transforms=None,batch_size=1,shuffle=False,num_workers=0):
        super(TestDataset,self).__init__(batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        img_list = list(glob.glob(os.path.join(img_dir,"*.jpg")))
        self.total_len = len(img_list)
        self.img_list = img_list
        self.transforms = transforms
    
    def __getitem__(self,index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        ori_img_size = img.size 
        if self.transforms is not None:
            img,_ = self.transforms(img,None)
        return img.astype(np.float32),ori_img_size,img_path.split("/")[-1].split(".jpg")[0]
    
    def collate_batch(self,batch):
        imgs = []
        img_sizes = []
        img_ids = []
        for img,img_size,ID in batch:
            imgs.append(img)
            img_sizes.append(img_size)
            img_ids.append(ID)
        imgs = np.stack(imgs,axis=0)
        return imgs,img_sizes,img_ids

    
def build_dataset(data_dir,anno_file,classnames,filter_empty=True,batch_size=1,shuffle=False,num_workers=0,is_train=False,use_all=False):
    train_imgs,test_imgs,all_imgs,classes = read_annotations(anno_file,filter_empty=filter_empty,classnames=classnames)
    if classnames is None:
        classnames = classes
        
    if is_train:
        annos = train_imgs
        transforms = train_transforms()
    else:
        annos = test_imgs
        transforms = val_transforms()
    if use_all:
        annos = all_imgs
        
    dataset = TrainDataset(data_dir,annos,classnames,transforms,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    return dataset

def build_testdataset(img_dir,batch_size=1,shuffle=False,num_workers=1):
    transforms = val_transforms()
    return TestDataset(img_dir,transforms=transforms,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
