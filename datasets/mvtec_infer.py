import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import glob

CLASS_NAMES = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut',
               'metal_nut', 'pill', 'screw', 'toothbrush',
               'transistor', 'zipper']

class MVTecDataset_infer(Dataset):
    def __init__(self, dataset_path='./mvtec_anomaly_detection', class_name=None,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x = self.load_dataset_folder_infer()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x

    def __len__(self):

        return len(self.x)
        

    def load_dataset_folder_infer(self):
        x = []
        inference_dir = os.path.join(self.dataset_path, self.class_name, 'inference')
        img_fpaths = sorted(glob.glob(os.path.join(inference_dir, '**/*.png'), recursive=True))
        x.extend(img_fpaths)
        return list(x)
        