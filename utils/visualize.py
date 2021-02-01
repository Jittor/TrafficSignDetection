import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import os

from utils.box_ops import calculate_ious

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 


def draw_boxes(img,boxes,labels,classnames,scores=None, color=(0,0,0)):
    if scores is None:
        scores = ['']*len(labels) 
    for box,score,label in zip(boxes,scores,labels):
        box = [int(i) for i in box]
        text = classnames[label-1]+(f': {score:.2f}' if not isinstance(score,str) else score)
        img = draw_box(img,box,text,color)
    return img

def visualize_result(img_file,
                     pred_boxes,
                     pred_scores,
                     pred_labels,
                     gt_boxes,
                     gt_labels,
                     classnames,
                     iou_thresh=0.5,
                     miss_color=(255,0,0),
                     wrong_color=(0,255,0),
                     surplus_color=(0,0,255),
                     right_color=(0,255,255)):
    
    img = cv2.imread(img_file)

    detected = [False for _ in range(len(gt_boxes))]
    miss_boxes = []
    wrong_boxes = []
    surplus_boxes = []
    right_boxes = []

    # sort the box by scores
    ind = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[ind,:]
    pred_scores = pred_scores[ind]
    pred_labels = pred_labels[ind]

    # add background
    classnames = ['background']+classnames

    for box,score,label in zip(pred_boxes,pred_scores,pred_labels):
        ioumax = 0.
        if len(gt_boxes)>0:
            ioumax,jmax = calculate_ious(gt_boxes,box)
        if ioumax>iou_thresh:
            if not detected[jmax]:
                detected[jmax]=True
                if label == gt_labels[jmax]:
                    right_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
                else:
                    wrong_boxes.append((box,f'{classnames[label]}->{classnames[gt_labels[jmax]]}'))
            else:
                surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
        else:
            surplus_boxes.append((box,f'{classnames[label]}:{int(score*100)}%'))
    
    for box,label,d in zip(gt_boxes,gt_labels,detected):
        if not d:
            miss_boxes.append((box,f'{classnames[label]}'))
    
    colors = [miss_color]*len(miss_boxes) + [wrong_color]*len(wrong_boxes) + [right_color]*len(right_boxes) + [surplus_color]*len(surplus_boxes)

    boxes = miss_boxes + wrong_boxes + right_boxes + surplus_boxes
    
    for (box,text),color in zip(boxes,colors):
        img = draw_box(img,box,text,color)
    
    # draw colors
    colors = [right_color,wrong_color,miss_color,surplus_color]
    texts = ['Detect Right','Detect Wrong Class','Missed Ground Truth','Surplus Detection']
    for i,(color,text) in enumerate(zip(colors,texts)):
        img = cv2.rectangle(img=img, pt1=(0,i*30), pt2=(60,(i+1)*30), color=color, thickness=-1)
        img = cv2.putText(img=img, text=text, org=(70,(i+1)*30-5), fontFace=0, fontScale=0.8, color=color, thickness=2)
    return img


def save_visualize_image(img_id,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames):
    img_file =  f'/mnt/disk/lxl/dataset/TTnew/anno/data/{img_id}.jpg'

    img = visualize_result(img_file,pred_boxes,pred_scores,pred_labels,gt_boxes,gt_labels,classnames)

    if not os.path.exists('test_imgs'):
        os.makedirs('test_imgs',exist_ok=True)

    cv2.imwrite(f'test_imgs/{img_id}.jpg',img)


def visualize_image(img_id,boxes,score,label,gt_boxes,gt_label,classnames,scale=(2048.0/800)):
    filename = f'/mnt/disk/lxl/dataset/tt100k/data/test/{img_id}.jpg'
    boxes = boxes*scale
    gt_boxes = gt_boxes*scale
    img = cv2.imread(filename)
    img = draw_boxes(img,boxes,label,classnames,score,color=(0,0,255))
    img = draw_boxes(img,gt_boxes,gt_label,classnames,color=(0,255,0))
    if not os.path.exists('test_imgs'):
        os.makedirs('test_imgs',exist_ok=True)
    cv2.imwrite(f'test_imgs/{img_id}.jpg',img)
