# LFVGS: Lightweight Few-shot View Gaussian Splatting reconstruction method

## Environmental Setups
Thanks to the project [Diff Gaussian Rasterization Depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth) for providing the depth correction rasterizer.
We provide install method based on Conda package and environment management:
```bash
cd submodules
unzip diff-gaussian-rasterization-depth.zip
cd ..
conda env create --file environment.yml
conda activate LVGS
mkdir checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
cd ..
```

## Data Preparation
For different datasets, divide the training set and use Colmap to extract point clouds based on the training views. Mip-NeRF 360 uniformly divides 24 views as input, while LLFF uses 3 views.Note that extracting point clouds using this method requires a GPU-supported version of Colmap.

``` 
cd LFSVGS
mkdir dataset 
cd dataset

# download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# run colmap to obtain initial point clouds with limited viewpoints
python tools/colmap_llff.py

# download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# run colmap on MipNeRF-360 dataset
python tools/colmap_360.py
```
If you encounter difficulties during data preprocessing, you can download dense point cloud data that has been preprocessed using Colmap. You may download them [through this link](https://drive.google.com/drive/folders/1VymLQAqzXtrd2CnWAFSJ0RTTnp25mLgA?usp=share_link). 

## Training
LLFF datasets. 
``` 
python train.py  -s dataset/nerf_llff_data/trex -m output/trex --eval --n_views 3 --comp --store_npz
```

MipNerf360 datasets
``` 
python train_360.py  -s dataset/mipnerf360/counter -s output/counter --eval --n_views 24 --comp --store_npz

If you need to evaluate on 9 views
Please remove the commented part of this code (https://github.com/LeeXiaoTong1/LFVGS/blob/24f1de9b99f2951953148a2e51e5c89f2dafc3b5/scene/dataset_readers.py#L263C1-L285C29). And comment out the line “ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")”

python train_360.py  -s dataset/mipnerf360/counter -s output/counter --eval --n_views 9 --comp --store_npz
```
For some scenarios, if the results are significantly different from those in the original paper, it is recommended to manually adjust the number of training rounds to 9,000.

## Rendering

```
python render.py -s dataset/nerf_llff_data/trex/ -m  output/trex --iteration {} 
```
If you want to obtain depth maps predicted by a monocular depth estimator.

```
python render.py -s dataset/nerf_llff_data/trex  -m  output/trex --iteration {} --render_depth
```


## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py -s dataset/nerf_llff_data/trex  -m  output/trex --iteration {}
```

## Acknowledgement

Our method benefits from these excellent works.
- [DNGaussian](https://github.com/Fictionarry/DNGaussian.git)
- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [SparseNeRF](https://github.com/Wanggcong/SparseNeRF)
- [Compact-3DGS](https://github.com/maincold2/Compact-3DGS)
- [MipNeRF-360](https://github.com/google-research/multinerf)
