# Coupling Self-Distillation with Test Time Augmentation for effective LiDAR-Based 3D Semantic Segmentation
This is the official implementation of "Coupling Self-Distillation with Test Time Augmentation for effective LiDAR-Based 3D Semantic Segmentation" paper, that you can download [here]().

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 18.04)
- Python 3.9.18
- PyTorch 1.10.0
- CUDA 11.1


## Install 

1. Construct an anaconda environment with python 3.9.18
2. Install pytorch 1.10 `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
3. Install [torchsparse](https://github.com/mit-han-lab/torchsparse) with `pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0`
4. For the k-NN, we use the operations as implemented in [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer). Execute the lib\pointops\setup.py file, downloaded from [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer),  with `python3.9 setup.py install` . The `pointops` module must be compiled with **NumPy < 2.0**, as NumPy 2.x introduces breaking changes for compiled extensions.  
If you see version-related errors, downgrade NumPy using:

```bash
pip install "numpy<2"
```

5. Install [h5py](https://docs.h5py.org/en/latest/build.html) with `conda install h5py`
6. Install tqdm with `pip install tqdm`
7. Install ignite with `pip install pytorch-ignite`
8. Install numba with `pip install numba`

## Supported Dataset
- [Street3D](https://kutao207.github.io/shrec2020)

### Street3D
- Plese follow the instructions from [here](https://kutao207.github.io/shrec2020) to download the Street3D dataset. It is in a `.txt` form. Place it in the `data/Street3D/txt` folder, where you should have two folders, `train` and `test` with 60 and 20 `.txt` files, respectively.
- Next, cd to the scripts/Street3D folder and execute the pre-processing scripts as follows:
 ```
 python street3d_txt_to_h5.py
 python street3d_partition_train.py
 python street3d_partition_test.py
 ```
 
 The first script converts the dataset to h5 format and places it in the `data/Street3D/h5` folder
 The following scripts split each scene into subscenes of around 80k points and save them in `.bin` format into proper folders, `train_part_80k` and `test_part_80k` sets, respectively. The `train_part_80k` folder should contain 2458 files and the `test_part_80k` folder should contain 845 files. Training and testing is performed based on these split subscenes of 80k points. 
 
 The final structure for both datasets should look like this:
 
 data/

 - Street3D/
   - txt/
     - train/
       - 5D4KVPBP.txt
       - ...
     - test/
       - 5D4KVPG4.txt
       - ...
    
   - h5/
     - train/
       - 5D4KVPBP.h5
       - ...
       
     - test/
       - 5D4KVPG4.h5
       - ...
       
     - train_part_80k/
       - 5D4KVPBP0.bin
       - ...
       
     - test_part_80k/
       - 5D4KVPG40.bin
       - ...
       
       
## Training

To train the networks, check the following script:
```
python scripts/Street3D/street3d_train_all.py
```
To train with Self Distillation and Test-Time Augmentation check the following script (in the scripts/Street3D folder):

```
python street3d_train_all_Self_Dist.py
```

Inside each file, you can select the proper network to train, as well as training parameters.


## Inference

To test the networks in Street3D test set with or without Test-Time Augmenation, check the following scripts (in the scripts/Street3D folder):
```
python street3d_inference_all.py
python street3d_inference_all_TTA.py
```
Inside each file, you can select the proper network to inference, as well as to load the proper weights.


## Pretrained weights



## Citation

