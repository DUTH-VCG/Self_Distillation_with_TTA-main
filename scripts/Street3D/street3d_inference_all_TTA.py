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
from meters.update_util import apply_tta_augmentations_just_points, custom_collate_grouped
import csv



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
#from models_street3d.minko_lfa import MinkUNet 
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
num_aug = 0
augmenation_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
model_name = "minko"

torch.set_printoptions(sci_mode=False)

class shrec_dataset:
    def __init__(self, voxel_size, source, train=True, split=None):
        self.source = source
        self.files = [os.path.join(source, f) for f in os.listdir(source)]
        self.voxel_size = voxel_size
        self.train = train
        self.split = split
        self.num_aug = num_aug


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        inputs = np.fromfile(path, np.float32).reshape(-1, 4)

        minpc = np.min(inputs[:, :3], axis=0, keepdims=True)
        coords = inputs[:, :3] - minpc
        feats = coords.copy()
        all_labels = inputs[:, -1]

        tta_inputs = []
        tta_inverses = []

        # Loop to generate num_aug augmentations
        if num_aug>0:
          for _ in range(num_aug):
              aug_coords = apply_tta_augmentations_just_points(coords)  # returns one augmented coords array

              aug_vox_coords, aug_indices, aug_inverse = sparse_quantize(
                  aug_coords, self.voxel_size, return_index=True, return_inverse=True
              )

              aug_input = SparseTensor(
                  coords=torch.tensor(aug_vox_coords, dtype=torch.int),
                  feats=torch.tensor(feats[aug_indices], dtype=torch.float)
              )

              tta_inputs.append(aug_input)
              tta_inverses.append(aug_inverse)
          return {
              'scene_id': index,
              'input': None,
              'tta_list': tta_inputs,
              'tta_inverses': tta_inverses,     
              'label': None,
              'all_labs': all_labels,
              'inverse': None,
              'minpc': minpc,
              'path': path
          }
        else:
          orig_coords, orig_indices, inverse = sparse_quantize(
              coords, self.voxel_size, return_index=True, return_inverse=True
          )

          orig_input = SparseTensor(
              coords=torch.tensor(orig_coords, dtype=torch.int),
              feats=torch.tensor(feats[orig_indices], dtype=torch.float)
          )
          orig_labels = torch.from_numpy(inputs[orig_indices, -1]).long()

          label_tensor = SparseTensor(
              coords=torch.tensor(orig_coords, dtype=torch.int),
              feats=orig_labels
          )

        return {
            'scene_id': index,
            'input': orig_input,
            'tta_list': [],
            'tta_inverses': [],     
            'label': label_tensor,
            'all_labs': all_labels,
            'inverse': inverse,
            'minpc': minpc,
            'path': path
        }





sourcetest = '../../data/Street3D/h5/test_part_80k'

# Initial voxel size (0.05m X 0.05m X 0.05m)
r = 0.05

# Batch size (keep this to 1 for inference)
bs = 1
print("Inference loop for " + str(model_name))
for aug in augmenation_list:
  num_aug = aug

  print(f"Starting inference for N = {num_aug}")

  torch.manual_seed(30)
  np.random.seed(30)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  datasettest = shrec_dataset(voxel_size=r,source=sourcetest,train=False)
  dltest = torch.utils.data.DataLoader(
      datasettest,
      batch_size=bs,
      collate_fn=custom_collate_grouped,
      num_workers=4
  )
  metricmiou = metric(num_classes=5)
  metricoa = metric(metric='overall',num_classes=5)



  model = MinkUNet(num_classes=5, cr=1,pres=1,vres=1,inc=3)


  model.cuda()

  model.load_state_dict(torch.load(modelpath),strict=False)


  prevfile=None
  data = []

  model.eval()

  # cmatrix = ConfusionMatrix(5,'recall')
  full_tensor = torch.zeros(845, 6)
  with torch.no_grad():
      all_aug_point_logits = []
      all_aug_labels = []
      for idx, test_dict in enumerate(tqdm(dltest)):
        if num_aug == 0:
          inputs = test_dict['input'].to('cuda')                     
          labels = test_dict['label'].to('cuda')                      
          inverse = test_dict['inverse']                            
          all_labs = test_dict['all_labs'].to('cuda') 
          outputs = model(inputs)
          outputs = outputs[inverse]
          metricmiou.update(outputs,all_labs.long())
          metricoa.update(outputs,all_labs.long())
        else:
          tta_list = test_dict['tta_list']                            
          tta_inverses = test_dict['tta_inverses'] 

          all_labs = test_dict['all_labs'].to('cuda') 
          aug_logits = []
          
          for aug_tensor, aug_inv in zip(tta_list[0], tta_inverses[0]):
                  aug_output = model(aug_tensor.to('cuda'))
                  aug_logits.append(aug_output[aug_inv])  
      
          avg_logits = torch.stack(aug_logits).mean(dim=0) 

          metricmiou.update(avg_logits,all_labs.long())
          metricoa.update(avg_logits,all_labs.long())
          # cmatrix.update((avg_logits.view(-1,5),all_labs.view(-1).int().cuda()))


  miou_savepath = "../../inference_outputs/"
  if not os.path.exists(miou_savepath):
    os.makedirs(miou_savepath)
  miou_savepath = "../../inference_outputs/" + model_name + f"_miou.txt"
  printed_string = f"Number of Augmenations is {num_aug}\n"
  print(printed_string)
  with open(miou_savepath, 'a') as f:
    f.write(printed_string + '\n')
  oa = metricoa.compute()
  miou = metricmiou.compute(True, savepath=miou_savepath)
  printed_string = f"testing oa is {oa:.4f} and miou is {miou:.4f}\n\n"
  with open(miou_savepath, 'a') as f:
    f.write(printed_string + '\n')
  print(printed_string)

  metricmiou.reset()
  metricoa.reset()
