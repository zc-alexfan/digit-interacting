# Getting Started

## Environment
DIGIT has been implemented and tested on Ubuntu 18.04 with python >= 3.7, PyTorch Lightning 0.9 and PyTorch 1.6.

Clone the repo:

```
git clone https://github.com/zc-alexfan/digit-interacting
```

Create folders needed:
```
make folders
```

Install conda environment:

```
conda create -n digit python=3.7
conda deactivate
conda activate digit
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```



## Downloading InterHand2.6M

- Download the 5fps.v1 of InterHand2.6M, following the instructions [here](https://github.com/facebookresearch/InterHand2.6M)
- Place annotations, images, and rootnet_output from [InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M) under `./data/InterHand/*`:


```
./data/InterHand
├── annotations
├── images
│   ├── test
│   ├── train
│   └── val
├── rootnet_output
│   ├── rootnet_interhand2.6m_output_all_test.json
│   └── rootnet_interhand2.6m_output_machine_annot_val.json
|-- annotations
|-- images
|   |-- test
|   |-- train
|   `-- val
`-- rootnet_output
    |-- rootnet_interhand2.6m_output_test.json
    `-- rootnet_interhand2.6m_output_val.json
```

- The folder `./data/InterHand/annotations` should look like this:

```
./data/InterHand/annotations
|-- skeleton.txt
|-- subject.txt
|-- test
|   |-- InterHand2.6M_test_MANO_NeuralAnnot.json
|   |-- InterHand2.6M_test_camera.json
|   |-- InterHand2.6M_test_data.json
|   `-- InterHand2.6M_test_joint_3d.json
|-- train
|   |-- InterHand2.6M_train_MANO_NeuralAnnot.json
|   |-- InterHand2.6M_train_camera.json
|   |-- InterHand2.6M_train_data.json
|   `-- InterHand2.6M_train_joint_3d.json
`-- val
    |-- InterHand2.6M_val_MANO_NeuralAnnot.json
    |-- InterHand2.6M_val_camera.json
    |-- InterHand2.6M_val_data.json
    `-- InterHand2.6M_val_joint_3d.json
```

## Preparing data and backbone for training

Download the ImageNet-pretrained backbone from [here](https://github.com/HRNet/HRNet-Human-Pose-Estimation) and place it under:

```
./saved_models/pytorch/imagenet/hrnet_w32-36af842e.pt
```

Package images into lmdb:

```
cd scripts
python package_images_lmdb.py
```

Preprocess annotation:

```
python preprocess_annot.py
```

Render part segmentation masks:

- Following the `README.md` of [render_mano_ih](https://github.com/zc-alexfan/render_mano_ih) to prepare an LMDB of part segmentation. For question in preparing the segmentation masks, please keep issues in there.

Place the LMDB from the images, the segmentation masks, and `meta_dict_*.pkl` to `./data/InterHand` and it should look like the structure below. The cache files `meta_dict_*.pkl` are by-products of the step above.

```
|-- annotations
|   |-- skeleton.txt
|   |-- subject.txt
|   |-- test
|   |   |-- InterHand2.6M_test_MANO_NeuralAnnot.json
|   |   |-- InterHand2.6M_test_camera.json
|   |   |-- InterHand2.6M_test_data.json
|   |   |-- InterHand2.6M_test_data.pkl
|   |   `-- InterHand2.6M_test_joint_3d.json
|   |-- train
|   |   |-- InterHand2.6M_train_MANO_NeuralAnnot.json
|   |   |-- InterHand2.6M_train_camera.json
|   |   |-- InterHand2.6M_train_data.json
|   |   |-- InterHand2.6M_train_data.pkl
|   |   `-- InterHand2.6M_train_joint_3d.json
|   `-- val
|       |-- InterHand2.6M_val_MANO_NeuralAnnot.json
|       |-- InterHand2.6M_val_camera.json
|       |-- InterHand2.6M_val_data.json
|       |-- InterHand2.6M_val_data.pkl
|       `-- InterHand2.6M_val_joint_3d.json
|-- cache
|   |-- meta_dict_test.pkl
|   |-- meta_dict_train.pkl
|   `-- meta_dict_val.pkl
|-- images
|   |-- test
|   |-- train
|   `-- val
|-- rootnet_output
|   |-- rootnet_interhand2.6m_output_test.json
|   `-- rootnet_interhand2.6m_output_val.json
`-- segm_32.lmdb
```
