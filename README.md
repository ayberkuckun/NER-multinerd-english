# NER-MultiNERD-English
Named Entity Recognition (NER) on the English subset of MultiNERD.

## How to run the code:
### Build the environment
- This repo is tested with `Python 3.9`.
```
python3.9 -m venv .venv
source .venv/bin/activate
```

### Install the requirements
- First install PyTorch by following the installation instructions in their
[Official website](https://pytorch.org/get-started/locally/). This repo is tested with 
`PyTorch v1.12.1` and `Conda 11.3`.

- Later, you can install the specific dependencies of this repo with:
```
pip install -r requirements.txt
```

### Set the Configurations
- Configurations for the experiment are located in the `configs.yaml`.

### Run the experiment
```
python experiment.py --config_path --log_to_wandb
```
- `--config_path`: Path to user defined configs.

- `--log_to_wandb` Allows logging the metrics to Weights & Biases. If you don't want to log the training
and evaluation values to wandb, remove the flag.
