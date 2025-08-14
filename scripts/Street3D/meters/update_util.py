import torch
import torch.nn.functional as F
from torchsparse.utils.collate import sparse_collate_fn, sparse_collate
import numpy as np
import numpy as np
import torch



def custom_collate_grouped(batch):
    batch_size = len(batch)

    # Collate student inputs
    student_batch = sparse_collate_fn([{'input': item['input']} for item in batch])['input']

    # Collate voxel labels
    label_batch = sparse_collate_fn([{'label': item['label']} for item in batch])['label']

    # TTA grouped augmentations and inverses

    tta_list_grouped = []
    tta_inverse_grouped = []
    if len(batch[0]['tta_list'])>0:
        for item in batch:
            scene_aug_list = []
            scene_inv_list = []
            for aug, inv in zip(item['tta_list'], item['tta_inverses']):
                aug_tensor = sparse_collate([aug])
                scene_aug_list.append(aug_tensor)
                scene_inv_list.append(inv)
            tta_list_grouped.append(scene_aug_list)
            tta_inverse_grouped.append(scene_inv_list)

    batch_inverses = [item['inverse'] for item in batch]
    batch_point_labels = [item['all_labs'] for item in batch]

    if batch_size == 1:
        if batch_inverses[0] is not None:
            inverse = torch.from_numpy(batch_inverses[0]).long()
        else:
            inverse = None
        all_labs = torch.from_numpy(batch_point_labels[0]).long()
    else:
        if batch_inverses[0] is not None:
            voxel_counts = [len(item['input'].coords) for item in batch]
            inverse_offsets = [0]
            for count in voxel_counts[:-1]:
                inverse_offsets.append(inverse_offsets[-1] + count)

            adjusted_inverses = [
                torch.from_numpy(inv).long() + offset
                for inv, offset in zip(batch_inverses, inverse_offsets)
            ]
            inverse = torch.cat(adjusted_inverses, dim=0)
        else:
            inverse = None
        all_labs = torch.cat(
            [torch.from_numpy(lab).long() for lab in batch_point_labels],
            dim=0
        )

    return {
        'input': student_batch,                      # Batched original scenes
        'tta_list': tta_list_grouped,                # List[List[SparseTensor]]
        'tta_inverses': tta_inverse_grouped,         # List[List[np.ndarray]]
        'label': label_batch,                        # Batched voxel labels
        'inverse': inverse,                          # Concatenated + offset-adjusted
        'all_labs': all_labs,                        # Concatenated
        'scene_id': [item['scene_id'] for item in batch],
        'minpc': [item['minpc'] for item in batch],
        'path': [item['path'] for item in batch]
    }


def apply_tta_augmentations_just_points(points: np.ndarray, ret_aug=False):
    """
    Apply compound TTA transformations in the order:
    scale → flip → rotate → translate

    Args:
        points (np.ndarray): (N, 3) array of point coordinates.
        num_augmentations (int): Number of augmentations to generate.

    Returns:
        List[np.ndarray]: List of transformed point clouds (N, 3)
    """
    pts = points.copy()

    # 1. Scale
    scale = np.random.uniform(0.95, 1.05)
    pts *= scale

    # 2. Flip along X or Y axis (randomly)
    flip_axis = np.random.choice([0, 1, None])
    if flip_axis is not None:
        pts[:, flip_axis] *= -1

    # 3. Rotate around Z axis
    theta = np.random.uniform(0, 2 * np.pi)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    pts = pts @ rot_mat.T

    # 4. Translate (Gaussian noise)
    translation = np.random.normal(0, 0.5, size=(1, 3))
    pts += translation
    tx, ty, tz = translation.flatten()

    if ret_aug:
        augmentations = {
            "scale": scale,
            "flip_axis": flip_axis,
            "rotation_theta": theta,
            "tx": tx,
            "ty": ty,
            "tz": tz,
        }
        return pts, augmentations
    else:

        return pts


def lovasz_softmax_flat(probs, labels, classes='present'):
    """
    probs: [P, C] Variable, class probabilities at each prediction (flattened)
    labels: [P] Tensor, ground truth labels (flattened)
    classes: 'all' or 'present'
    """
    if probs.numel() == 0:
        return probs * 0.  # empty tensor

    C = probs.size(1)
    losses = []

    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        loss = torch.dot(errors_sorted, grad)
        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0., device=probs.device)
    return torch.mean(torch.stack(losses))

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    """
    gts = gt_sorted.sum()
    if gts == 0:
        return torch.zeros_like(gt_sorted)

    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax(logits, labels, classes='present'):
    """
    probs: [B, P, C] Tensor, class probabilities
    labels: [B, P] Tensor, ground truth labels
    """
    if logits.dim() == 2:  # [P, C]
        logits = logits.unsqueeze(0) 

    if labels.dim() == 1:  # [P]
        labels = labels.unsqueeze(0) 
    
    probs = F.softmax(logits, dim=-1)
    B = probs.size(0)
    losses = []
    for b in range(B):
        loss = lovasz_softmax_flat(probs[b], labels[b], classes=classes)
        losses.append(loss)
    return torch.mean(torch.stack(losses))
