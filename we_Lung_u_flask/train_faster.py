import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import pandas as pd
import collections
from tqdm import tqdm

"""
faster rcnn+mask rcnn 으로 구성
둘의 차이? faster rcnn은 object detection으로 끝나지만, mask rcnn은 instance segmentation까지 수행!
즉, faster rcnn은 물체가 있는 위치에 네모 박스만 쳤지만 mask rcnn은 물체의 mask를 따낸다.
따라서, object detection까지는 faster rcnn을 수행하고 instance segmentation을 위해 mask rcnn을 수행
<<two stage>>
ROI(물체가 있을지도 모르는 위치의 후보 영역) 제안 -> ROI에 대해 클래스 분류 및 bbox 회귀
따라서, 느리지만 성능은 좋음
"""

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model
 
class OpenDataset(torch.utils.data.Dataset):
# 데이터셋을 생성하고 Dataloader로 데이터셋을 불러오는 클래스
# height와 width는 resize할 크기, transforms는 이미지 전처리(좌우 변환 등)를 의미
    def __init__(self, root, height, width, transforms=None):
        self.root = root
        self.transforms = transforms
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)

        lines = []
        with open(filename, 'r') as f:
            csvreader = csv.reader(f)
            for line in csvreader:
                lines.append(line)

        lines = lines[1:] # remove csv headers
        counter = 0
        for i in lines:
            filename, minX, maxX, minY, maxY, classname = i
            self.image_info[counter]['filename'] = filename
            self.image_info[counter]['box'] = [float(minX),float(maxX),float(minY),float(maxY)]
            # 0은 background를 의미
            if classname is 'covid-19' : self.image_info[counter]['classname'] = 1
            elif classname is 'nodule' : self.image_info[counter]['classname'] = 2
            elif classname is 'cancer' : self.image_info[counter]['classname'] = 3
            counter += 1

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.image_info[idx]['filename'])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        info = self.image_info[idx]
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        boxes = torch.as_tensor([info['box']], dtype=torch.float32)
        labels = torch.as_tensor(info['classname'], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)

ImageFile.LOAD_TRUNCATED_IMAGES = True

num_classes = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_train = OpenDataset("../input/train/", "../input/train.csv", 128, 128, transforms=None)
 
# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=4, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)
 
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
 
# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

num_epochs = 10
 
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    #evaluate(model, data_loader_test, device=device)

torch.save(model.state_dict(), "model.pth")