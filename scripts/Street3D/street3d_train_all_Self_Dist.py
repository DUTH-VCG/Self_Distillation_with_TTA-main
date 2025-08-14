import random
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn
import os
import copy

from meters.s3dis import MeterS3DIS as metric
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from functools import partial
from core.schedulers import cosine_schedule_with_warmup
import math
from meters.update_util import apply_tta_augmentations_just_points, lovasz_softmax, custom_collate_grouped


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

## Define the proper savepath
model_name = "minko"
model_type = "Minko_Models/"
savepath = "../../Street3D_results/"+model_type+model_name
## Define the proper training parameters
## Voxel size in meters (0.05 equals to a voxel of 0.05 X 0.05 X 0.05)
r = 0.05

## Batch Size
bs = 2

## Number of Epochs
ne = 15

## Number of Augmenations
num_aug = 10

## Learning rate is set to optimizer below

##-----------------------------------------------------------------------------


def calculate_weights(labels):
  counts = np.zeros(5)
  labels = labels.cpu().numpy()
  for i in range(5):
    counts[i]+=np.sum(labels==i)
  frq = counts/np.sum(counts)
  return 1/(frq+0.00001)**0.5



random.seed(2341)
np.random.seed(2341)
torch.manual_seed(2341)
torch.cuda.manual_seed_all(2341)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print(f"the folder {savepath} exists change the savepath and try again")
    exit(0)

if not os.path.exists(savepath+f'/worse'):
    os.makedirs(savepath+f'/worse')
if not os.path.exists(savepath+f'/teacher'):
    os.makedirs(savepath+f'/teacher')


class shrec_dataset:
    def __init__(self, voxel_size, source, train=True, split=None):
        self.source = source
        self.files = [os.path.join(source, f) for f in os.listdir(source)]
        self.voxel_size = voxel_size
        self.train = train
        self.split = split
        self.num_aug = num_aug

        # if self.train:
        #     random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        inputs = np.fromfile(path, np.float32).reshape(-1, 4)

        minpc = np.min(inputs[:, :3], axis=0, keepdims=True)
        coords = inputs[:, :3] - minpc
        
        all_labels = inputs[:, -1]
        
        
        tta_inputs = []
        tta_inverses = []

        for _ in range(num_aug):
            aug_coords = apply_tta_augmentations_just_points(coords)  # returns one augmented coords array
            feats = aug_coords.copy()
            aug_vox_coords, aug_indices, aug_inverse = sparse_quantize(
                aug_coords, self.voxel_size, return_index=True, return_inverse=True
            )

            aug_input = SparseTensor(
                coords=torch.tensor(aug_vox_coords, dtype=torch.int),
                feats=torch.tensor(feats[aug_indices], dtype=torch.float)
            )

            tta_inputs.append(aug_input)
            tta_inverses.append(aug_inverse)

        theta = np.random.uniform(0,2*np.pi)
        scale = np.random.uniform(0.95,1.05)
        rot_mat = np.array([[np.cos(theta),np.sin(theta),0],
                            [-np.sin(theta), np.cos(theta), 0],
                            [0,0,1]])
        coords[:,:3] = np.dot(inputs[:,:3],rot_mat)*scale
        feats = coords.copy()
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
            'tta_list': tta_inputs,
            'tta_inverses': tta_inverses,     
            'label': label_tensor,
            'all_labs': all_labels,
            'inverse': inverse,
            'minpc': minpc,
            'path': path
        }



source = '../../data/Street3D/h5/train_part_80k'

files = os.listdir(source)
random.shuffle(files)
sourcetrain = files


ftrain = open(savepath+"/train.txt",'w')

## Set which dataset to use
dataset = shrec_dataset(voxel_size=r,source=source,split='train_part')

metricmiou = metric(num_classes=5)
metricoa = metric(metric='overall',num_classes=5)
t_metricmiou = metric(num_classes=5)


dataflow = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=bs,
    collate_fn=custom_collate_grouped,
    num_workers=1,
)



model = MinkUNet(num_classes=5, cr=1,pres=1,vres=1,inc=3)
t_model = MinkUNet(num_classes=5, cr=1,pres=1,vres=1,inc=3)

model.cuda()
t_model.cuda()

criterion = nn.CrossEntropyLoss()
t_criterion = nn.CrossEntropyLoss()
bestmiou=-1
bestoa=0

optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,nesterov=True,weight_decay=1.0e-4,lr=0.024)



scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=ne,
                              batch_size=bs,
                              dataset_size=len(dataset)))


check = 0
prev_miou = 0


for epoch in range(ne):
    model.train()
    t_model.eval()

    metricmiou.reset()
    metricoa.reset()

    for idx, feed_dict in enumerate(tqdm(dataflow)):
        inputs = feed_dict['input'].to('cuda')                     
        labels = feed_dict['label'].to('cuda')                      
        tta_list = feed_dict['tta_list']                            
        tta_inverses = feed_dict['tta_inverses']                  
        inverse = feed_dict['inverse']                            
        all_labs = feed_dict['all_labs'].to('cuda')  

        gt_labels_combined = all_labs.long()
        avg_teacher_logits = []

        for scene_idx in range(len(tta_list)):
            aug_logits = []
            for aug_tensor, aug_inv in zip(tta_list[scene_idx], tta_inverses[scene_idx]):
                with torch.no_grad():
                    aug_output = t_model(aug_tensor.to('cuda'))
                    aug_logits.append(aug_output[aug_inv])  

            scene_avg = torch.stack(aug_logits).mean(dim=0) 
            avg_teacher_logits.append(scene_avg)
     
        class_weights = torch.from_numpy(calculate_weights(gt_labels_combined)).cuda().float()   
        criterion.weight = class_weights    
        teacher_logits_combined = torch.cat(avg_teacher_logits, dim=0)       

        student_output = model(inputs)                                      
        student_point_logits = student_output[inverse]    
        


        hard_loss = criterion(student_point_logits, gt_labels_combined) + \
                   lovasz_softmax(student_point_logits, gt_labels_combined)
        t_metricmiou.reset()
        t_metricmiou.update(teacher_logits_combined, gt_labels_combined)
        t_miou = t_metricmiou.compute(False)


        soft_loss = t_criterion(
            student_point_logits,
            torch.argmax(teacher_logits_combined.detach(), dim=1)
        )

        gamma = min(2.718, math.exp(t_miou)/2) 

        total_loss = hard_loss + gamma * soft_loss 

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        metricmiou.update(student_point_logits, gt_labels_combined)
        metricoa.update(student_point_logits, gt_labels_combined)
        t_model = copy.deepcopy(model)
        t_model.eval()  
        for param in t_model.parameters():
            param.requires_grad = False

    miou = metricmiou.compute(True)
    oa = metricoa.compute()
    print(f"epoch is {epoch} oa is {oa} and miou is {miou}")
    print(f"{epoch} {oa} {miou}",file=ftrain)
    metricmiou.reset()
    metricoa.reset()
    torch.save(t_model.state_dict(),savepath+'/teacher/shrec.pth')
    if bestmiou < miou:
        bestmiou = miou
        torch.save(model.state_dict(),savepath+'/shrec.pth')
    else: 
        print("Lower miou than previously)")
        torch.save(model.state_dict(),savepath+'/worse/shrec.pth')
    prev_miou = miou

