# NER-MultiNERD-English
Named Entity Recognition (NER) on the English subset of MultiNERD.

## How to run the code:
### Build the environment
This repository is tested with `Python 3.9`. To set up the environment, execute the following commands:
```
python3.9 -m venv .venv
source .venv/bin/activate
```

### Install the requirements
- Before proceeding, make sure to install PyTorch by following the installation instructions on their
[Official website](https://pytorch.org/get-started/locally/). This repository is tested with
`PyTorch v1.12.1` and `CUDA 11.3`.

- Afterward, install the specific dependencies of this repository with:
```
pip install -r requirements.txt
```

### Set the Configurations
- Configuration settings for the experiment are stored in the `configs.yaml` file.

### Run the experiment
Execute the following command to run the experiment:
```
python experiment.py --config_path --log_to_wandb
```
- `--config_path`: Path to user-defined configurations.
- `--log_to_wandb` Allows logging metrics to Weights & Biases. If you prefer not to log training and 
evaluation values to wandb, you can remove this flag.
