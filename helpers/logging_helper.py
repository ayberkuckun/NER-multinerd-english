import os

import yaml


def set_wandb_logging_env(file_path):
    """
    Sets the API key and the username for wandb logging.
    """
    with open(file_path) as file:
        wandb_env_vars = yaml.safe_load(file)

    os.environ["WANDB_API_KEY"] = wandb_env_vars["WANDB_API_KEY"]
    os.environ["WANDB_ENTITY"] = wandb_env_vars["WANDB_ENTITY"]
