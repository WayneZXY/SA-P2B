# SA-P2B

## Introduction

This repository is released for SA-P2B. Here we include our SA-P2B model (PyTorch) and code for data preparation, training and testing on KITTI tracking dataset.

## Preliminary

* Install ``python 3.6``.

* Install dependencies.

```bash
    pip install -r requirements.txt
```

* Build `_ext` module.
  
```bash
    python setup.py build_ext --inplace
```

* Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

* Download [velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), [calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and [label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip) in the dataset and place them under the same parent folder.

## Evaluation

Train a new P2B model on KITTI data:

```bash
python train_tracking.py --data_dir=<kitti data path>
```

Test a new P2B model on KITTI data:

```bash
python test_tracking.py --data_dir=<kitti data path>
```

Please refer to the code for setting of other optional arguments, including data split, training and testing parameters, etc.

## Acknowledgements

Thank Haozhe Qi for his implementation of [P2B](https://github.com/HaozheQi/P2B).
Thank Chenhang He for his implementation of [SA-SSD](https://github.com/skyhehe123/SA-SSD).
Thank Giancola for his implementation of [SC3D](https://github.com/SilvioGiancola/ShapeCompletion3DTracking).
Thank Erik Wijmans for his implementation of [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) in PyTorch.
Thank Charles R. Qi for his implementation of [Votenet](https://github.com/facebookresearch/votenet).
They help and inspire this work.
