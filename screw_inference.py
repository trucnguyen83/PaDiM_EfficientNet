#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. import module
## utils
import random
import os
from random import sample
import numpy as np
import pickle
import time
import sys
import copy
import argparse
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, precision_recall_curve
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import seaborn as sns


## torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

## eff model
from efficient_modified import EfficientNetModified

## mvtec datasets
import datasets.mvtec_infer as mvtec_infer

## filter warnings
warnings.filterwarnings('ignore')


# In[2]:


# 2. choose device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"PyTorch Version: {torch.__version__}")
print("GPU is", "available" if torch.cuda.is_available() else "NOT AVAILABLE")


# In[3]:


# 3. functions
# def parse_args():
#     parser = argparse.ArgumentParser('PaDiM Parameters')
#     parser.add_argument('-d', '--data_path', type=str, required=True, help='mvtec data location')
#     parser.add_argument('-s', '--save_path', type=str, required=True, help='inference model & data location')
#     parser.add_argument('-a', '--arch', type=str, choices=['b0', 'b1', 'b4', 'b7'], default='b4')
#     parser.add_argument('-b', '--batch_size', type=int, default=32)
#     parser.add_argument('--training', action='store_true')
#     parser.add_argument('--seed', type=int, default=1024)
#     parser.add_argument('--resize', type=int, default=256)
#     parser.add_argument('--cropsize', type=int, default=224)
#     parser.add_argument('--model_print', action='store_true')
#     parser.add_argument('--img_print', action='store_true')
    
#     return parser.parse_args()
    # epoch, random_select size

def create_seed(filters):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()

    s = int(H1/H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)

    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def show_feat_list(model, size=(1, 3, 224, 224)):
    sample_inputs = torch.zeros(size)
    feat_list = model.extract_entire_features(sample_inputs)
    for i, feat in enumerate(feat_list, 0):
        print(i, feat.shape)

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return img.mul_(std).add_(mean)    

def calc_covinv(embedding_vectors, H, W, C):
      for i in range(H * W):
        yield np.linalg.inv(np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * np.identity(C))

def plot_fig_infer(infer_img, scores, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() # or 1
    vmin = scores.min() # or 0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num):
        img = infer_img[i]
        img = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8) # denormalize
        
        file_path = infer_dataset.x[i]  # Get the file path
        file_name = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))
        new_name = folder_name +'/'+ file_name
        
        heat_map = scores[i]
        normalized_heat_map = (heat_map - vmin) / (vmax - vmin)
                
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
#if morphology.disk(*) value too high, when score slightly > threshold, boundary may not be drawn, esp when area is small 
        kernel = morphology.disk(1) 
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(right=0.9)

        predicted_label = (img_scores[i] > img_threshold).astype(int)

        title = 'Sample {}\nImg_threshold: {:.3f}, Img_score: {:.3f}, Predicted: {}, Pixel_threshold: {:.3f}'.format(
            new_name, img_threshold, img_scores[i], predicted_label, threshold)
        fig_img.suptitle(title)
        
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(img, cmap='gray', interpolation = 'none')
        ax_img[1].imshow(normalized_heat_map, cmap='jet', alpha=0.5, interpolation = 'none', vmin = 0, vmax =1)
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'), cax=cbar_ax)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        
       
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


# In[4]:


def get_args(data_path, save_path, arch='b4', batch_size=32, training=False, seed=1024, resize=256, cropsize=224, model_print=False, img_print=False):
    class Args:
        pass

    args = Args()
    args.data_path = data_path
    args.save_path = save_path
    args.arch = arch
    args.batch_size = batch_size
    args.training = training
    args.seed = seed
    args.resize = resize
    args.cropsize = cropsize
    args.model_print = model_print
    args.img_print = img_print

    return args

if __name__ == '__main__':
    data_path = './mvtec_anomaly_detection'
    save_path = 'result'
    arch = 'b4' #can be b0,b1,b4,b7
    batch_size = 32
    training = True
    seed = 1024
    resize = 256
    cropsize = 224
    model_print = True
    img_print = True
    
    args = get_args(data_path, save_path, arch, batch_size, training, seed, resize, cropsize, model_print, img_print)

    name = 'efficientnet-{}'.format(args.arch)
    eff_model = EfficientNetModified.from_pretrained(name)

    if args.model_print:
        print(eff_model)


# In[5]:


# make directory for saving data
os.makedirs(os.path.join(args.save_path, 'model_pkl_%s' % name), exist_ok=True)


if args.arch == 'b0':
    block_num = torch.tensor([3, 5, 11]) # b0 
    filters = (24 + 40 + 112) # 176
elif args.arch == 'b1':
    # block_num = torch.tensor([3, 6, 9]) # b1 first, 24 + 40 + 80
    # block_num = torch.tensor([4, 7, 13]) # b1 medium 24 + 40 + 112
    block_num = torch.tensor([5, 8, 16]) # b1 last 24 + 40 + 112
    filters = (24 + 40 + 112) # 176
elif args.arch == 'b4':
    # block_num = torch.tensor([3, 7, 11]) # b4 (32 + 56 + 112)
    block_num = torch.tensor([3, 7, 17]) # b4 (32 + 56 + 160)
    # block_num = torch.tensor([5, 9, 13]) # (32 + 56 + 112)
    # block_num = torch.tensor([5, 9, 20]) # b4 (32 + 56 + 160)
    # block_num = torch.tensor([6, 10, 22]) # b4 (32 + 56 + 160)
    filters = (32 + 56 + 160) # 248
elif args.arch == 'b7':
    block_num = torch.tensor([11, 18, 38]) # b7 (48 + 80 + 224) # last
    # block_num = torch.tensor([5, 12, 29]) # b7 (48 + 80 + 224) # first
    # block_num = torch.tensor([8, 15, 33]) # medium
    filters = (48 + 80 + 224) # 352

'''
The number of filters is so small that I want to take the entire filter, not randomly. 
So I'm going to delete the random code this time.
'''
create_seed(filters)

# model attach to device
eff_model.to(device)


# In[7]:


class_name = 'screw'
with open('result/model_pkl_efficientnet-b4/train_screw.pkl', 'rb') as f: #input your own path
    train_outputs = pickle.load(f)

mean = torch.Tensor(train_outputs[0]).to(device)
cov_inv = torch.Tensor(train_outputs[1]).to(device)

# Load the model
eff_model.eval()
    
infer_dataset = mvtec_infer.MVTecDataset_infer(args.data_path, class_name=class_name)
infer_dataloader = DataLoader(infer_dataset, batch_size=args.batch_size, pin_memory=True)
infer_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
print('Number of inference samples: ', len(infer_dataset))

infer_imgs = []

inference_start = time.time()

for x in tqdm(infer_dataloader, '| feature extraction | inference | %s |' % class_name):
    infer_imgs.extend(x.cpu().detach().numpy())

    with torch.no_grad():
        feats = eff_model.extract_features(x.to(device), block_num.to(device))

    for k, v in zip(infer_outputs.keys(), feats):
        infer_outputs[k].append(v.cpu().detach())

for k, v in infer_outputs.items():
    infer_outputs[k] = torch.cat(v, 0)

embedding_vectors = infer_outputs['layer1']
for layer_name in ['layer2', 'layer3']:
    embedding_vectors = embedding_concat(embedding_vectors, infer_outputs[layer_name])

B, C, H, W = embedding_vectors.size()
embedding_vectors = embedding_vectors.view(B, C, H * W).to(device)

dist_list = torch.zeros(size=(H*W, B))
for i in range(H*W):
    delta = embedding_vectors[:, :, i] - mean[:, i]
    m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[:, :, i]), delta.t())))
    dist_list[i] = m_dist

dist_list = dist_list.transpose(1, 0).view(B, H, W)
score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear', align_corners=False).squeeze().cpu().numpy()

for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)

inference_time = (time.time() - inference_start) / len(infer_dataset)
print('{} inference time per sample: {:.3f}'.format(class_name, inference_time))

# Normalization
# max_score = score_map.max()
# min_score = score_map.min()
max_score =  212.04628 #hard code from test data with anomaly
min_score =  12.293375 #hard code from test data with anomaly
scores = (score_map - min_score) / (max_score - min_score)

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1) #max of all pixel scores


# Hard code the threshold values
img_threshold = 0.45
threshold = 0.45


save_dir_infer = args.save_path + '/' + f'pictures_efficientnet_inference-{args.arch}'
os.makedirs(save_dir_infer, exist_ok=True)
#plot test images and detection
plot_fig_infer(infer_imgs, scores, threshold, save_dir_infer, class_name)


# In[ ]:




