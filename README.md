# NAC Framework
This repo is the official implementation of "Do Not Train It: A Linear Neural Architecture Search of Graph Neural Networks" (Xu et al., ICML 2023)


## Introduction
The originization of this repo is shown as follow:
```
|-- README.md # short introduction of codes
|-- nac # the python implementation of this project, including NAS Searching Phase and Finetuning Phase
        |-- __init__.py
        |-- controller # NAS updating modules
        |-- lr_scheduler
        |-- model
        |-- optimizer
        |-- solver
        `-- utils
|-- configs  # the typical usage configuration
`-- examples # the configuration for runing experiments
```

## Enviroment
First, install the required enviromental setup.
```bash
conda create -n nac python=3.7
conda activate nac

# please change the cuda/device version as you need
pip install -r requirements.txt
```

## Usage
Go to the workspace dir of examples,

1. First, change the ROOT Path in scripts to the current dir of the repo:
```bash
ROOT=/mnt/home/pxu22/codes/NAC -> Current Path to Repo
export PYTHONPATH=$ROOT:$PYTHONPATH
```

2. Run the train script first to get the searched result:
```bash
bash train.sh
```

3. Go to the finetune dir, and get the finetuned result of specific architecture
```bash
cd Finetune;
bash finetune.sh;
```

## Citation
```
@inproceedings{xu2023do,
  title={Do Not Train It: A Linear Neural Architecture Search of Graph Neural Networks},
  author={Peng Xu and Lin Zhang and Xuanzhou Liu and Jiaqi Sun and Yue Zhao and Haiqin Yang and Bei Yu},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```