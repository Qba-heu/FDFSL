## Cross-domain Few-shot Learning Based on Feature Disentanglement for Hyperspectral Image Classification

This repository includes the `PyTorch` implementation for the paper [FDFSL](https://ieeexplore.ieee.org/document/10494577).
```
@ARTICLE{10494577,
  author={Qin, Boao and Feng, Shou and Zhao, Chunhui and Li, Wei and Tao, Ran and Xiang, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Cross-Domain Few-Shot Learning Based on Feature Disentanglement for Hyperspectral Image Classification}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2024.3386256}}
```

This repo is a modification on the [DCFSL](https://github.com/Li-ZK/DCFSL-2021). Installation and details follow that repo.

## Requirements
* CUDA = 11.3

* Python = 3.8 

* Pytorch = 1.9 

* sklearn = 0.23.2

* numpy = 1.19.2


## Usage:
- [x] The preprocessing of source dataset refer to [DCFSL](https://github.com/Li-ZK/DCFSL-2021).
- [x] Runing the FDFSL on IndianPines dataset by `python FDFSL.py -da IndianPines`.
- [x] You can obtain the results for all three datasets by running the shell file `sh run_FDFSL.sh`.
- [x] The output floder is `./Results/out_{dataset}1-5.log`.