import multiprocessing

import yaml


def get_configs(config_file):
    """
    Gets the user defined configs from a YAML file and defines rest of the configs.
    """
    try:
        with open(config_file) as file:
            configs = yaml.safe_load(file)

    except:
        raise FileNotFoundError("Please give the correct path to the configs YAML file.")

    configs["systemA"]["id2label"] = {id: label for label, id in configs["systemA"]["label2id"].items()}
    configs["systemB"]["id2label"] = {id: label for label, id in configs["systemB"]["label2id"].items()}

    configs["num_proc"] = _auto_multiprocessing_core_count() if configs["multiprocess"] == "auto" else configs[
        "multiprocess"]

    if type(configs["num_proc"]) is not int or configs["num_proc"] < 1:
        raise ValueError("Please provide a positive integer to define the number of cores to be used.")

    return configs


def _auto_multiprocessing_core_count():
    """
    Returns the number of cores in the machine for multiprocessing.
    """
    return multiprocessing.cpu_count()
