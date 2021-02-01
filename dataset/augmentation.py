import numpy as np 
import json
import os
import cv2 
import glob

MAX_SIZE = 160
MIN_SIZE = 30
MAX_RATIO = 1.4
MIN_RATIO = 0.6
MAX_POSITION = 2048-240
AUG_NUM=150

def read_anno(file):
    if os.path.exists(file):
        return json.load(open(file))
    return []

def overlaps(bbox_a,bbox_b):
    # top left
    tl = np.maximum(bbox_a[:, None,:2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:,None,2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:,None] + area_b - area_i)

def filter_ious(boxes,iou_thresh=0.1):
    if boxes.shape[0]==0:
        return False
    ious = overlaps(boxes,boxes)
    lx = [i for i in range(ious.shape[0])]
    ious[lx,lx]=0
    fails = np.sum(ious>iou_thresh)
    if fails>0:
        return True
    return False

def _random_boxes(num_boxes):
    x1,y1 = np.random.uniform(0,MAX_POSITION,size=(num_boxes,1)),np.random.uniform(0,MAX_POSITION,size=(num_boxes,1))
    w = np.random.uniform(MIN_SIZE,MAX_SIZE,size=(num_boxes,1))
    h = w*np.random.uniform(MIN_RATIO,MAX_RATIO,size=(num_boxes,1))
    boxes = np.concatenate([x1,y1,x1+w,y1+h],axis=1)
    return boxes

def random_boxes(num_boxes,num_images,num_classes):
    boxes = _random_boxes(num_boxes) 
    img_indexes = np.random.randint(0,num_images,size=(num_boxes,))
    counts = 0
    while counts<100:
        aug_classes = np.random.randint(0,num_classes,size=(num_boxes,))
        _,counts = np.unique(aug_classes,return_counts=True)
        counts = np.min(counts)

    img_ids = np.unique(img_indexes)
    for i in img_ids:
        r = True
        while r:
            img_boxes = boxes[img_indexes==i,:]
            n = img_boxes.shape[0]
            r = filter_ious(img_boxes)
            if r:
                boxes[img_indexes==i,:] = _random_boxes(n)
    
    return boxes,img_indexes,aug_classes

def add_noise(img,sigma=0.02):
    noise = np.random.randn(img.shape[0],img.shape[1],1)*np.max(img)
    img = img.astype(np.float32)
    img = img+noise*sigma
    img = np.maximum(img,0)
    img = np.minimum(img,255)
    img = img.astype(np.uint8)
    return img

def paste_mark(img_file,boxes,classes,need_aug,mark_images,save_file,is_add_noise=True):
    img = cv2.imread(img_file)
    for b,c in zip(boxes,classes):
        mark = mark_images[need_aug[c]]
        w,h = int(b[2]-b[0]),int(b[3]-b[1])
        x,y = int(b[0]),int(b[1])
        mark = cv2.resize(mark,(w,h))
        if mark.shape[2]==4:
            ratio = (mark[:,:,3:]>0)*1.0
            mark = mark[:,:,:3]
        else:
            ratio = np.ones((mark.shape[0],mark.shape[1],1))
        if is_add_noise:
            mark = add_noise(mark)
        img[y:y+h,x:x+w]=mark*ratio+img[y:y+h,x:x+w]*(1.0-ratio)
    cv2.imwrite(save_file,img)

def find_dir(data_dir,img_id):
    t_f = f"{data_dir}/test/{img_id}.jpg"
    tt_f = f"{data_dir}/train/{img_id}.jpg"
    o_f = f"{data_dir}/other/{img_id}.jpg"
    if os.path.exists(tt_f):
        return f"train/{img_id}.jpg"
    elif os.path.exists(t_f):
        return f"test/{img_id}.jpg"
    elif os.path.exists(o_f):
        return f"other/{img_id}.jpg"
    assert False,f"{img_id}.jpg is not exists"

def select_empty_images(other_dir,annos):
    other_files = list(glob.glob(f"{other_dir}/*.jpg"))
    empty_files = []
    for f in other_files:
        img_id = f.split("/")[-1].split(".jpg")[0]
        if img_id not in annos or len(annos[img_id]["objects"])==0:
            empty_files.append(f)
    return empty_files
            
def build_annotations():
    data_dir = "/data/lxl/dataset/tt100k/tt100k_2021"
    mark_dir = f"{data_dir}/crop_marks"
    remark_dir = f"{data_dir}/re_marks"
    anno_f = f"{data_dir}/annotations_all.json"
    
    crop_marks(data_dir)
    data = json.load(open(anno_f))
    annos = data["imgs"]
    classnames = data["types"]
    empty_files = select_empty_images(f"{data_dir}/other",annos)

    marks = {}
    for category in classnames:
        m_f = f"{mark_dir}/{category}.png"
        rm_f = f"{remark_dir}/{category}.jpg"
        a = os.path.exists(m_f)
        marks[category]= m_f if a else rm_f
        if not os.path.exists(marks[category]):
            assert False,category

    category_nums = {c:0 for c in classnames}    
    for img_id,v in annos.items():
        objects = v["objects"]
        for o in objects:
            category = o["category"]
            category_nums[category]+=1

    need_aug = [c for c,v in category_nums.items() if v<100]
    mark_images = {c:cv2.imread(marks[c],cv2.IMREAD_UNCHANGED) for c in need_aug}
    
    num_boxes = len(need_aug)*AUG_NUM
    num_images = len(empty_files)
    num_classes = len(need_aug)
    boxes,img_indexes,aug_classes = random_boxes(num_boxes,num_images,num_classes)

    count = 0 
    annotations = {"imgs":{},"types":classnames}

    for img_i in np.unique(img_indexes):
        img_boxes = boxes[img_indexes==img_i,:]
        img_file = empty_files[img_i]
        file_name = img_file.split("/")[-1]
        img_id = file_name.split(".")[0]

        anno = {"path":f"augmentations/{file_name}","id":int(img_id)}
        classes = aug_classes[img_indexes==img_i]

        os.makedirs(f"{data_dir}/augmentations",exist_ok=True)
        save_file = img_file.replace("other","augmentations")
        paste_mark(img_file,img_boxes,classes,need_aug,mark_images,save_file)

        objects = []
        for box,c in zip(img_boxes,classes):
            bbox = {"xmin":float(box[0]),"ymin":float(box[1]),"xmax":float(box[2]),"ymax":float(box[3])}
            objects.append({"bbox":bbox,"category":need_aug[c]})
        anno["objects"]=objects
        annotations["imgs"][img_id]=anno 

        count+=1
        print(count,"/",len(empty_files))
    
    for img_id,v in annotations["imgs"].items():
        if img_id not in annos:
            annos[img_id]=v
        else:
            assert len(annos[img_id]["objects"])==0
            annos[img_id] = v

    json.dump({"imgs":annos,"types":classnames},open(f"{data_dir}/annotations_aug.json","w"))
    
def crop_marks(data_dir):
    img_files = glob.glob(f"{data_dir}/marks/*.png")
    for img_file in img_files:
        img = cv2.imread(img_file,cv2.IMREAD_UNCHANGED)
        alpha = img[:,:,3]
        ww = np.sum(alpha,axis=0)
        l = 0
        while ww[l]==0:
            l+=1
        r = ww.shape[0]-1
        while ww[r-1]==0:
            r-=1
    
        hh = np.sum(alpha,axis=1)

        t = 0
        while hh[t]==0:
            t+=1
        b = hh.shape[0]-1
        while hh[b-1]==0:
            b-=1
        img = img[t:b,l:r]
        os.makedirs(f"{data_dir}/crop_marks",exist_ok=True)
        cv2.imwrite(img_file.replace("marks","crop_marks"),img)

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 


def draw_boxes(img,boxes,classnames,color=(255,0,0)):
    for box,label in zip(boxes,classnames):
        box = [int(i) for i in box]
        img = draw_box(img,box,label,color)
    return img

def test():
    data_dir = "/data/lxl/dataset/tt100k/tt100k_2021"
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir,exist_ok=True)
    anno_f = data_dir+"/annotations_aug.json"
    annos = json.load(open(anno_f))
    imgs = list(annos["imgs"].values())
    imgs = [img for img in imgs if "augmentations" in img["path"]]
    indexes = [i for i in range(len(imgs))]
    np.random.shuffle(indexes)
    for i in indexes[:10]:
        data = imgs[i]
        path = data_dir+"/"+data["path"]
        objects = data["objects"]
        img = cv2.imread(path)
        boxes = [[o['bbox']['xmin'],o['bbox']['ymin'],o['bbox']['xmax'],o['bbox']['ymax']] for o in objects]
        labels = [o['category'] for o in objects]
        img = draw_boxes(img,boxes,labels)
        img_file = f"{tmp_dir}/{data['id']}.jpg"
        cv2.imwrite(img_file,img)

def main():
    np.random.seed(0)
    build_annotations()
    test()

if __name__ == '__main__':
    main()