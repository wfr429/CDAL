# CDAL-OSMA
This folder contains the implementation of the OSMA experiments.

Our implementation is based on the Pytorch version code of [POSE](https://github.com/ICTMCG/POSE).

## Prerequisites
- Linux
- NVIDIA GPU + CUDA 11.1
- Python 3.7.13
- PyTorch 1.10.0

## Dataset
The whole dataset is hosted [here](https://zenodo.org/record/8103474). Download, unzip, and put the dataset into the directory ``./dataset/``.

The annotation files are in ``./dataset`` and can be downloaded along with the project. The organization is as follows: 
  ```
  dataset
  ├── $split{id}_test
  │   └── annotations
  │       ├── $split{id}_test.txt
  │       ├── $split{id}_test_out.txt
  │       ├── $split{id}_test_out_seed.txt
  │       ├── $split{id}_test_out_arch.txt
  │       └── $split{id}_test_out_dataset.txt
  ├── $split{id}_train
  │   └── annotations
  │       └── $split{id}_train.txt
  └── $split{id}_val
      └── annotations
          └── $split{id}_val.txt
  ```
where `split{id}_train.txt, split{id}_val.txt, split{id}_test.txt` are the annotation files for training, validation, closed-set testing. `split{id}_test_out.txt` is the annotation file for all open-set/unknown data. `split{id}_test_out_seed.txt, split{id}_test_out_arch.txt, split{id}_test_out_dataset.txt` are annotation files for unseen seed, unseen architecture, and unseen dataset respectively. 


## Training
  - Run the following script:
  ```
  sh ./script/run_train.sh
  ```
## Testing

  - Run the following script:
  ```
  sh ./script/run_test.sh
  ```
