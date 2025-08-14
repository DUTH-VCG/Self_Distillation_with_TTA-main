from typing import Any, Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
import h5py

from meters.s3dis import MeterS3DIS as metric
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from ignite.metrics.confusion_matrix import ConfusionMatrix


## Uncomment to load the baseline model
from core.models import MinkUNet

## or the proper attention enhanced baseline model (only one model should be used at a time)
# from models_street3d.minko_se_global import MinkUNet 
# from models_street3d.minko_cbam_global import MinkUNet 
# from models_street3d.minko_knn_se import MinkUNet 
# from models_street3d.minko_knn_se_fast import MinkUNet 
# from models_street3d.minko_knn_cbam import MinkUNet 
# from models_street3d.minko_knn_cbam_fast import MinkUNet 
# from models_street3d.minko_pt import MinkUNet 
# from models_street3d.minko_pt_fast import MinkUNet 
# from models_street3d.minko_lfa import MinkUNet 
# from models_street3d.minko_lfa_fast import MinkUNet 

## Use the 'fast' implementations only for inference, after the network is trained

## Load the proper weights for each model
modelpath = '../../pretrained_weights/normal_training/minko/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_se/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_cbam/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_knnse/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_knncbam/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_pt/shrec.pth'
#modelpath = '../../pretrained_weights/normal_training/minko_lfa/shrec.pth'


## Load the proper weights for each model trained with self distillation
# modelpath = '../../pretrained_weights/self_distillation/minko_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_se_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_cbam_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_knnse_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_knncbam_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_pt_self_dist/shrec.pth'
#modelpath = '../../pretrained_weights/self_distillation/minko_lfa_self_dist/shrec.pth'


##----------------------------------------------------------------------------------
##-----------------------------------------------------------------------------



torch.set_printoptions(sci_mode=False)

class shrec_dataset:

    def __init__(self,voxel_size,source,train=True) :
        self.data = []
        self.train = train
        for file in os.listdir(source):
          path = os.path.join(source,file)
          self.data.append(path)
        self.voxel_size = voxel_size

    def __getitem__(self, item) -> Dict[str, Any]:

        path = self.data[item]
        inputs = np.fromfile(path,np.float32).reshape(-1,4)
        minpc = np.min(inputs[:,:3], axis=0, keepdims=True)
        inputs[:,:3] -= np.min(inputs[:,:3], axis=0, keepdims=True)
        coords, feats = inputs[:,:3], inputs[:,:3]
        coords, indices, inverse_mapping = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True,
                                          return_inverse=True)

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        all_labels = inputs[:,-1]
        labels = torch.from_numpy(inputs[indices,-1]).long()

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)

        return {'input': input,
#                'label': label}
                'label': label,
                'all_labs': all_labels,
                'inverse': inverse_mapping,
                'minpc': minpc,
                'path': path }

    def __len__(self):
        return len(self.data)
#        return 1




sourcetest = '../../data/Street3D/h5/test_part_80k'

# Initial voxel size (0.05m X 0.05m X 0.05m)
r = 0.05

# Batch size (keep this to 1 for inference)
bs = 1

datasettest = shrec_dataset(voxel_size=r,source=sourcetest,train=False)
dltest = torch.utils.data.DataLoader(
    datasettest,
    batch_size=bs,
    collate_fn=sparse_collate_fn,
    num_workers=4
)
metricmiou = metric(num_classes=5)
metricoa = metric(metric='overall',num_classes=5)



model = MinkUNet(num_classes=5, cr=1,pres=1,vres=1,inc=3)


model.cuda()

model.load_state_dict(torch.load(modelpath),strict=False)


print(sum(p.numel() for p in model.parameters() if p.requires_grad))


prevfile=None
data = []

model.eval()

# cmatrix = ConfusionMatrix(5,'recall')

with torch.no_grad():
    for test_dict in tqdm(dltest):
      inputs = test_dict['input'].to('cuda')
      labels = test_dict['label'].to('cuda')
      all_lab = test_dict['all_labs']
      inverse = test_dict['inverse']
      path = test_dict['path']
      minpc = test_dict['minpc']
      file = path[0].split(".")[-2].split("/")[-1]
      outputs = model(inputs)
      outputs = outputs[inverse]
      metricmiou.update(outputs.permute(0,2,1),all_lab.cuda())
      metricoa.update(outputs.permute(0,2,1),all_lab.cuda())
    #   cmatrix.update((outputs.view(-1,5),all_lab.view(-1).int().cuda()))


oa = metricoa.compute()
miou = metricmiou.compute(True)
print(f" testing oa is {oa} and miou is {miou}")

# print ("\n Confusion matrix is:")
# print(cmatrix.compute())
metricmiou.reset()
metricoa.reset()
