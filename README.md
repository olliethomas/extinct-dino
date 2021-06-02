# extinct-dino 🦕 ☄️

## Pre-requisites
1. [Poetry](https://python-poetry.org/docs/#installation) (check with `which poetry`)
2. [Conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) (recommended, not strictly required)
3. Python 3.9

## Setup
1. Clone the repo `git clone <link to repo>`
2. `cd extinct-dino`
3. `conda create -n ed python=3.9`
4. `conda activate ed`
5. `poetry install`
6. `poetry build`
7. `poetry install` <- This extra install is required the first time. For reasons.
8. If you're installing on a GPU enabled device then you must also do the following:
    1. `pip uninstall torch torchvision`
    2. [Install PyTorch with a CUDA version](https://pytorch.org/get-started/locally/) that's suitable for your machine. (e.g. at Sussex `pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`)

## Running
Now everthing is installed, you can run the project with `rawr`.

To see the options: `rawr --help`

For more information on the `Trainer` args, see [pytorch-lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags)


