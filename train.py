#coding=utf-8
import jittor as jt 
import numpy as np 
from tqdm import tqdm
from jittor import optim
import argparse
import sys
import glob
import pickle
import os
from tensorboardX import SummaryWriter

from utils.ap_eval import calculate_VOC_mAP
from utils.visualize import save_visualize_image
from dataset.data import build_dataset
from model.faster_rcnn import FasterRCNN

DATA_DIR = '/data/lxl/dataset/tt100k/tt100k_2021'
CLASSNAMES = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'i6', 'i7', 'i8', 'i9', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pd', 'pe', 'pnl', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pm2.5', 'ph4.4', 'ph3.3', 'ph2.6', 'i4l', 'i2r', 'im', 'wc', 'pcr', 'pcl', 'pss', 'pbp', 'p1n', 'pbm', 'pt', 'pn-2', 'pclr', 'pcs', 'pcd', 'iz', 'pmb', 'pdd', 'pctl', 'ph1.8', 'pnlc', 'pmblr', 'phclr', 'phcs', 'pmr']
EPOCHS = 10
save_checkpoint_path = f"{DATA_DIR}/checkpoints"

def eval(val_dataset,faster_rcnn,epoch,is_display=False,is_save_result=True,score_thresh=0.01):
    faster_rcnn.eval()
    results = []
    for batch_idx,(images,boxes,labels,image_sizes,img_ids) in tqdm(enumerate(val_dataset)):
        result = faster_rcnn.predict(images,score_thresh=score_thresh)
        for i in range(len(img_ids)):
            pred_boxes,pred_scores,pred_labels = result[i]
            gt_boxes = boxes[i]
            gt_labels = labels[i]
            img_id = img_ids[i]
            results.append((img_id.item(),pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy(),gt_boxes.numpy(),gt_labels.numpy()))
            if is_display:
                save_visualize_image(DATA_DIR,img_id,pred_boxes.numpy(),pred_scores.numpy(),pred_labels.numpy(),gt_boxes.numpy(),gt_labels.numpy(),CLASSNAMES)
    if is_save_result:
        os.makedirs(save_checkpoint_path,exist_ok=True)
        pickle.dump(results,open(f"{save_checkpoint_path}/result_{epoch}.pkl","wb"))
    mAP,_ = calculate_VOC_mAP(results,CLASSNAMES,use_07_metric=False)
    return mAP


def test():
    val_dataset = build_dataset(data_dir=DATA_DIR,
                                anno_file=f'{DATA_DIR}/annotations_aug.json',
                                classnames=CLASSNAMES,
                                batch_size=1,
                                shuffle=False,
                                num_workers=8,
                                is_train=False)
    faster_rcnn = FasterRCNN(n_class=len(CLASSNAMES)+1)
    files = sorted(list(glob.glob(f'{save_checkpoint_path}/checkpoint*.pkl')))
    f = files[-1]
    faster_rcnn.load(f)           
    mAP = eval(val_dataset,faster_rcnn,0,is_display=True,is_save_result=False,score_thresh=0.5)
    print(mAP)
        
def train():
    train_dataset = build_dataset(data_dir=DATA_DIR,
                                  anno_file=f'{DATA_DIR}/annotations_aug.json',
                                  classnames=CLASSNAMES,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=4,
                                  is_train=True,
                                  use_all=True)
    
    val_dataset = build_dataset(data_dir=DATA_DIR,
                                anno_file=f'{DATA_DIR}/annotations_aug.json',
                                classnames=CLASSNAMES,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                is_train=False,
                                use_all=True)
    
    faster_rcnn = FasterRCNN(n_class = len(CLASSNAMES)+1)

    optimizer = optim.SGD(faster_rcnn.parameters(),momentum=0.9,lr=0.001)
    
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        faster_rcnn.train()
        dataset_len  = len(train_dataset)
        for batch_idx,(images,boxes,labels,image_sizes,img_ids) in tqdm(enumerate(train_dataset)):
            rpn_loc_loss,rpn_cls_loss,roi_loc_loss,roi_cls_loss,total_loss = faster_rcnn(images,boxes,labels)
            
            optimizer.step(total_loss)

            writer.add_scalar('rpn_cls_loss', rpn_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('rpn_loc_loss', rpn_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_loc_loss', roi_loc_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('roi_cls_loss', roi_cls_loss.item(), global_step=dataset_len*epoch+batch_idx)
            writer.add_scalar('total_loss', total_loss.item(), global_step=dataset_len*epoch+batch_idx)
            
            if batch_idx % 10 == 0:
                loss_str = '\nrpn_loc_loss: %s \nrpn_cls_loss: %s \nroi_loc_loss: %s \nroi_cls_loss: %s \ntotoal_loss: %s \n'
                print(loss_str % (rpn_loc_loss.item(),rpn_cls_loss.item(),roi_loc_loss.item(),roi_cls_loss.item(),total_loss.item()))
        
        mAP = eval(val_dataset,faster_rcnn,epoch)
        writer.add_scalar('map', mAP, global_step=epoch)
        os.makedirs(save_checkpoint_path,exist_ok=True)
        faster_rcnn.save(f"{save_checkpoint_path}/checkpoint_{epoch}.pkl")

def main():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    parser.add_argument('--task',help='Task(train,test)',default='test',type=str)
    parser.add_argument('--no_cuda', help='not use cuda', action='store_true')
    args = parser.parse_args()
    
    if not args.no_cuda:
        jt.flags.use_cuda=1

    if args.task == 'test':
        test()
    elif args.task == 'train':
        train()
    else:
        print(f"No this task: {args.task}")

if __name__ == '__main__':
    main()
    

    
    
