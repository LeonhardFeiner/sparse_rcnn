# Sparse Convolutions for Semantic 3D Instance Segmentation
## Abstract

3D object detection and segmentation are crucial for various domains and applications. However, transferring 2D image techniques to 3D data is still challenging because of the massive amount of data contained in 3D voxel grids. We present an architecture, which combines the principle of object detection and segmentation used by [Mask R-CNN](https://arxiv.org/abs/1703.06870)  for 2D images with the computational eﬃciency of [Sparse Submanifold Convolutions](https://arxiv.org/abs/1706.01307) on sparse 3D voxel grids. The network consists of a Region Proposal Network to predict bounding boxes and both a Class Network and a Mask Network which rely on the region proposals. We show how parts of the feature extractor, the Class Network and the Mask Network can be rendered sparse. A sparse feature extractor reduces the amount of required computation while keeping similar detection performance. A sparse Mask Network enables to process masks of diﬀerent shapes batch-wise without resizing and loosing spatial correspondence information. Furthermore, we propose a solution to ﬁnd the best density of anchors by using anchor-wise anisotropic anchor densities with respect to each anchor’s shape. Our model proves that the Mask R-CNN based 3D model can achieve both state-of-the-art object detection and instance segmentation performance.

## Results
The  method has been evaluated on the [Scannet Benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=369)

## Getting Started
### Prerequirements
This setup is tested on Ubuntu 18.04 with CUDA 10.1. Furthermore it requires [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to be installed.
### Installation
1. Download this git repository
```
git clone git@github.com:LeonhardFeiner/sparse_rcnn.git
```
2. create an Anaconda environment using the environment file of this repo
```
cd sparse_rcnn/
conda env create -f environment.yml
conda activate py38_pt14_scn
```
3. Download the [SparseConvNet repository](https://github.com/facebookresearch/SparseConvNet)
```
cd ..
git clone git@github.com:facebookresearch/SparseConvNet.git
```
4. Install SparseConvNet
```
cd SparseConvNet/
bash develop.sh
```
5. Install [Meshlab](http://www.meshlab.net/) for data preparation

### Dataset
1. Download the [Scannet Dataset](http://www.scan-net.org/)

Required Files:
 * `_vh_clean.aggregation.json`
 * `_vh_clean_2.ply` 
 * `_vh_clean_2.0.010000.segs.json`
 
Optional high resolution data (not recommended):
 * `_vh_clean.ply`
 * `_vh_clean.segs.json`
2. Adapt the paths in `scannet_config\pathes.py`
3. Correct error in "scene0217_00" by running 
```
python preparation\0_raw_data_error_correction.py
```
4. Create label mapper by running 
```
python preparation\1_sparse_label_map.py
```
5. Create the instance association tensors by running 
```
python preparation\2_sparse_instance_association.py
```
5. Calculate vertex normals using meshlab by running 
```
python preparation\3_sparse_normals.py
```
6. Calculate the dataset by 
```
python preparation\4_sparse_create_data.py
```
7. Precalculate augmented data (not required and not recommended)
```
python preparation\5_precalculate_dataset.py
```

### Start Training
Parameters of the network can be adapted here:
```
scannet_config\run.py
```
The training can be started with
```
python main.py
```
## Acknowledgments
I'd like to thank Prof. Dr. Matthias Nießner for sharing his ideas and supervising my work.
