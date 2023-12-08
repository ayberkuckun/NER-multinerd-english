import argparse

import wandb

from helpers import training_helper, configs_helper, dataset_helper, model_helper, logging_helper


def experiment(config_path, log_to_wandb):
    """
    Main function for the experiment.

    :param: config_path: Path to the user defined configs.
    :param: log_to_wandb: True if logging to the Wandb is desired.
    """
    # Get configs.
    configs = configs_helper.get_configs(config_path)

    # Prepare Tokenizer and Data Collator. Both systems use the same tokenizer.
    tokenizer = model_helper.get_tokenizer(configs=configs, system="systemA")
    data_collator = model_helper.get_data_collator(tokenizer=tokenizer)

    # Prepare datasets.
    systemA_dataset = dataset_helper.get_dataset(
        tokenizer=tokenizer,
        configs=configs,
        system="systemA",
        load_ds_from_disk=configs["systemA"]["load_ds_from_disk"],
        preprocess=configs["systemA"]["preprocess"],
    )
    systemB_dataset = dataset_helper.get_dataset(
        tokenizer=tokenizer,
        configs=configs,
        system="systemB",
        load_ds_from_disk=configs["systemB"]["load_ds_from_disk"],
        preprocess=configs["systemB"]["preprocess"],
    )

    # Prepare models.
    systemA_model = model_helper.get_model(configs=configs, system="systemA")
    systemB_model = model_helper.get_model(configs=configs, system="systemB")

    # Start Logging for system A.
    if log_to_wandb:
        configs["training_arguments"]["report_to"] = "wandb"

        # Prepare for logging.
        logging_helper.set_wandb_logging_env(configs["wandb_env_var_path"])

        wandb.login()
        wandb.init(
            project=configs["wandb_project_name"],
            name=f"systemA_{configs['dataset_name']}-{configs['systemA']['model_checkpoint']}"
        )

    # Finetune system A.
    systemA_trainer = training_helper.get_trainer(
        model=systemA_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset=systemA_dataset,
        configs=configs,
        system="systemA"
    )
    systemA_trainer.train()

    # Evaluate system A. We transform system B dataset in order to evaluate on the common NER categories.
    sysytemA_test_set = systemB_dataset['test'].map(
        dataset_helper.get_prepare_system_a_test_set_fn(configs), num_proc=configs["num_proc"]
    )
    systemA_metrics = systemA_trainer.evaluate(sysytemA_test_set, metric_key_prefix="test")

    # Overall precision value needs to be recalculated.
    systemA_metrics = training_helper.fix_system_a_metrics(systemA_metrics, log_to_wandb)
    print(systemA_metrics)

    # Start Logging for system B.
    if log_to_wandb:
        wandb.finish()
        wandb.init(
            project=configs["wandb_project_name"],
            name=f"systemB_{configs['dataset_name']}-{configs['systemB']['model_checkpoint']}"
        )

    # Finetune system B.
    systemB_trainer = training_helper.get_trainer(
        model=systemB_model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset=systemB_dataset,
        configs=configs,
        system="systemB"
    )
    systemB_trainer.train()

    # Evaluate system B.
    systemB_metrics = systemB_trainer.evaluate(systemB_dataset['test'], metric_key_prefix="test")
    print(systemB_metrics)

    # Finish Logging.
    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Argument parsing
    arguments_parser = argparse.ArgumentParser(description="Run the Exp.")
    arguments_parser.add_argument("--config_path", required=True, help="Path to user defined configs", type=str)
    arguments_parser.add_argument(
        "--log_to_wandb", help="Allows logging the metrics to Weights & Biases.", action="store_true"
    )
    args = arguments_parser.parse_args()

    experiment(args.config_path, args.log_to_wandb)
