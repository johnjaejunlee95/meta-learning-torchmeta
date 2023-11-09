# Meta-Learning Implementation

Implementation of **MAML** and **Prototypical Network** using torchmeta (additional methods will be added, soon.)


## Dependency

First, **download the torchmeta** package:

```
pip install torchmeta
```
or
```
git clone https://github.com/tristandeleu/pytorch-meta.git
cd pytorch-meta
python setup.py install
```



## Datasets: miniImageNet, tieredImageNet, CIFAR-FS, CUB

Next, **download the datasets** (reference from https://github.com/renmengye/few-shot-ssl-public):

**`miniImageNet`**: [[Google Drive Link]](https://drive.google.com/file/d/1ihY4yuw0PNq1q7PzvF-oYovSuQ_Rco9t/view "As a result of modifications to Google Drive's policy, downloading will now need to be done manually.") 

**`tieredImageNet`**: [[Google Drive Link]](https://drive.google.com/file/d/1-0ptcP7Rbnrhex-mEUxTmImz1jRnXe6L/view "As a result of modifications to Google Drive's policy, downloading will now need to be done manually.")

**`CIFAR_FS`**: [[Google Drive Link]](https://drive.google.com/file/d/1--SLwRqQzIRu_RcK91L4UjrGR7y267FN/view "As a result of modifications to Google Drive's policy, downloading will now need to be done manually.") (or download available via code $\rightarrow$ highly recommend)

**`CUB`** :  [[Google Drive Link]](https://drive.google.com/file/d/1DzHXVq1A1OPvwR5XezvouC2U6vrFNW4N/view "As a result of modifications to Google Drive's policy, downloading will now need to be done manually.")

### Check `datasets.py`:

- In the **`datasets.py`** file located within the **`utils`** folder, you will come across the import statements for datasets such as miniImagenet and tieredImagenet. When you navigate to the imported file, it's necessary to modify `torchmeta.datasets.utils` $\Rightarrow$ `torchvision.datasets.utils` because of **check integretity**.
- Make sure to verify that the `download`  is set to `True`  in the **`datasets.py`** (**`CIFAR_FS`** is available)
  $\Rightarrow$ As a result of modifications to Google Drive's policy, downloading will now need to be done manually. Move the datasets as indicated in the provided datasets path below.
- `DATA_PATH`  : Your own datasets folder path 

```
DATA_PATH
└─cub
  | CUB_200_2011.tgz
└─cifar100
  | cifar-fs
  | data.hdf5
  | fine_names.json
└─miniimagenet
  | mini-imagenet.tar.gz
└─tieredimagenet
  | tiered-imagenet.tar
```



## Training / Evaluation / Test

### Training

#### MAML

```
python train_maml.py --datasets [DATASETS] --epoch 60000 --num_shots 5 --batch_size 2 
```
#### Prototypical Networks

`Euclidean` classifier
```
python train_proto.py --datasets [DATASETS] --num_ways_proto 20 --num_shots 5 --epoch 200 --batch_size 100 
```


### Evaluation / Test

```
python eval_meta.py --[OPTIONS]
```



```
option arguments:  
--epoch:              epoch number (default: 60000)  
--num_ways:           N-way (default: 5)  
--num_ways_proto:     N-way for Proto-Net (default: 30)  
--num_shots:          k shots for support set (default: 5)  
--num_shots_test:     number of query set (default: 15) 
--imgc:               RGB(image channel) (default: 3)  
--filter_size:        size of convolution filters (default: 64)  
--batch_size:         meta-batch size (default: 2)  
--max_test_task:      number of tasks for evaluation (default: 1000)  
--meta_lr:            outer-loop learning rate (default: 1e-3)  
--update_lr:          inner-loop learning rate (default: 1e-2)  
--update_step:        number of inner-loop update steps while training (default: 5)  
--update_test_step:   number of inner-loop update steps while evaluating (default: 10) 
--update:             update method: MAML, ANIL, BOIL (default: MAML)
--scale_factor:       Scaling factor for the cosine classifier (default: 10)
--dropout:            dropout probability (default: 0.2)
--gpu_id:             gpu device number (default: 0)
--model:              model architecture: Conv-4, ResNet12 (default: conv4)
--datasets:           datasets: miniimagenet, tieredimagenet, cifar-fs, CUB (default: miniimagenet)
--version:            file version (default: 0)  
```


## Result

### MAML
| Datasets                                                               |          |   5 ways - 1 shot    |   5 ways - 5 shot    |
|------------------------------------------------------------------------|----------|:--------------------:|:--------------------:|
| **`mini-ImageNet`** ([MAML](https://arxiv.org/pdf/1703.03400.pdf))     | Original |   48.70 $\pm$ 1.84   |   63.11 $\pm$ 0.92   |
|                                                                        | **Ours** | **48.79 $\pm$ 0.16** | **62.43 $\pm$ 0.89** |
| **`tiered-ImageNet`** ([TPN](https://arxiv.org/pdf/1805.10002.pdf))    | Original |   52.54 $\pm$ 0.35   |   70.97 $\pm$ 0.51   |
|                                                                        | **Ours** | **50.01 $\pm$ 0.28** | **65.58 $\pm$ 0.15** |
| **`CIFAR_FS`** ([R2-D2](https://arxiv.org/pdf/1805.08136.pdf))         | Original |   58.90 $\pm$ 1.90   |   71.50 $\pm$ 1.00   |
|                                                                        | **Ours** | **57.36 $\pm$ 0.37** | **72.41 $\pm$ 0.78** |
| **`CUB`** ([FEAT](https://arxiv.org/pdf/1812.03664.pdf))               | Original |   55.92 $\pm$ 0.95   |   72.09 $\pm$ 0.76   |
|                                                                        | **Ours** | **56.98 $\pm$ 0.28** | **73.64 $\pm$ 0.23** |

### Prototypical Network (Higher-Way)
`Euclidean`
| Datasets                                                               |          |    5 ways - 1 shot   |    5 ways - 5 shot   |
|------------------------------------------------------------------------|----------|:--------------------:|:--------------------:|
| **`mini-ImageNet`** ([ProtoNet](https://arxiv.org/pdf/1703.05175.pdf)) | Original |   49.42 $\pm$ 0.78   |   68.20 $\pm$ 0.66   |
|                                                                        | **Ours** | **49.45 $\pm$ 0.23** | **66.17 $\pm$ 0.15** |
| **`tiered-ImageNet`** ([TPN](https://arxiv.org/pdf/1805.10002.pdf))    | Original |   53.31 $\pm$ 0.89   |   72.69 $\pm$ 0.74   |
|                                                                        | **Ours** | **52.54 $\pm$ 0.35** | **71.97 $\pm$ 0.51** |
| **`CIFAR_FS`** ([R2-D2](https://arxiv.org/pdf/1805.08136.pdf))         | Original |   55.50 $\pm$ 0.70   |   72.00 $\pm$ 0.60   |
|                                                                        | **Ours** | **54.33 $\pm$ 0.20** | **73.60 $\pm$ 0.19** |
| **`CUB`** ([FEAT](https://arxiv.org/pdf/1812.03664.pdf))               | Original |   51.31 $\pm$ 0.91   |   70.77 $\pm$ 0.69   |
|                                                                        | **Ours** | **51.13 $\pm$ 0.77** | **70.23 $\pm$ 0.81** |


## Reference

- [torchmeta](https://github.com/tristandeleu/pytorch-meta)
- [SiMT](https://github.com/jihoontack/SiMT/tree/main) 
- [BOIL](https://github.com/HJ-Yoo/BOIL)
# meta-learning-pytorch
