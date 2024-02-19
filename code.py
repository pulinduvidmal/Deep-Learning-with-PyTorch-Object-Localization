# install libraries/packages/modules

!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install timm
!pip install --upgrade opencv-contrib-python

# Download Dataset

!git clone https://github.com/parth1620/object-localization-dataset.git

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('/content/object-localization-dataset')

# Configurations

csv_file="/content/object-localization-dataset/train.csv"
data_dir = "/content/object-localization-dataset/"
DEVICE ='cuda'
batch_size=16
img_size = 140
LR = 0.001
EPOCHS = 40
MODEL_NAME ='efficientnet b0 '
NUM_COR =4

df=pd.read_csv(csv_file)

# Understand the dataset

row =df.iloc[111]
img = cv2.imread(data_dir + row.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pt1 = (row.xmin,row.ymin)
pt2 = (row.xmax,row.ymax)
bnd_box_IMG = cv2.rectangle(img,pt1,pt2,(255,0,0),2)

train_df,valid_df=train_test_split(df,test_size =0.2 , random_state =42)

# Augmentations

import albumentations as A

train_augs=A.Compose([
    A.Resize(img_size,img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(),
],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))
val_augs=A.Compose([
    A.Resize(img_size,img_size),
],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels']))

# Create Custom Dataset

class ObjLocDataset(torch.utils.data.Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax

        bbox = [[xmin, ymin, xmax, ymax]]

        img_path = data_dir + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            data = self.augmentations(image=img, bboxes=bbox, class_labels=[None])
            img = data['image']
            bbox = data['bboxes'][0]

        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0
        bbox = torch.Tensor(bbox)

        return img, bbox

trainset = ObjLocDataset(train_df, train_augs)
validset = ObjLocDataset(valid_df, val_augs)

# Load dataset into batches

trainloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
validloader = torch.utils.data.DataLoader(validset,batch_size=batch_size,shuffle=False)

# Create Model

from torch import nn
import timm

class OLModel(nn.Module):
    def __init__(self):
        super(OLModel,self).__init__()
        self.backbone = timm.create_model('efficientnet_b0',pretrained= True, num_classes = 4)

    def forward(self,images,gt_bboxes = None):
        bboxes = self.backbone(images)

        if gt_bboxes!= None:
            loss=nn.MSELoss()(bboxes,gt_bboxes)
            return bboxes,loss
        return bboxes

model = OLModel()

# Training Loop

optimizer=torch.optim.Adam(model.parameters(),lr=LR)

import numpy as np

best_valid_loss = np.Inf

for i in range(EPOCHS):
    train_loss = train_fn(model, trainloader, optimizer)
    valid_loss = eval_fn(model, validloader)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best_model.pt')
        print("WEIGHTS ARE SAVED")
        best_valid_loss = valid_loss
    print(f"epoch:{i+1}, train_loss:{train_loss}, val_loss:{valid_loss}")

# Inference

import utils

model.load_state_dict(torch.load('best_model.pt'))
model.eval()

with torch.no_grad():
    image, gt_bbox = valset[12]
    image = image.unsqueeze(0)
    out_bbox = model(image)

    utils.compare_plots(image, gt_bbox, out_bbox)
