import numpy as np
import evaluate
import wandb
from transformers import TrainingArguments, Trainer


def get_trainer(model, tokenizer, data_collator, dataset, configs, system):
    """
    Gets the trainer object for the desired system.
    """
    trainer = Trainer(model=model,
                      args=_get_training_arguments(configs, system),
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['validation'],
                      data_collator=data_collator,
                      compute_metrics=_get_evaluation_fn(configs, system),
                      tokenizer=tokenizer)

    return trainer


def _get_training_arguments(configs, system):
    """
    Prepares the training arguments.
    """
    training_args = TrainingArguments(output_dir=configs[system]["output_dir"], **configs["training_arguments"])

    return training_args


def _get_evaluation_fn(configs, system):
    """
    Getter of compute_metrics function.
    """
    seqeval = evaluate.load('seqeval')

    def compute_metrics(logits_and_labels):
        """
        Evaluation function to compute metrics.
        """
        logits, labels = logits_and_labels

        predictions = np.argmax(logits, axis=-1)

        # Remove labels and predictions for tags that are not part of the sentence.
        true_labels = [[configs[system]["id2label"][l] for l in label if l != -100] for label in labels]

        true_predictions = [
            [
                configs[system]["id2label"][p] for p, l in zip(prediction, label) if l != -100
            ] for prediction, label in zip(predictions, labels)
        ]

        all_metrics = seqeval.compute(predictions=true_predictions, references=true_labels)

        return all_metrics

    return compute_metrics


def fix_system_a_metrics(metrics, log_to_wandb):
    """
    Fixes the overall precision/f1 metrics using only the mutual classes.
    """
    total_support = metrics["test_PER"]["number"] + metrics["test_ORG"]["number"] + metrics["test_LOC"]["number"] \
                    + metrics["test_ANIM"]["number"] + metrics["test_DIS"]["number"]

    metrics["test_overall_precision"] = (
            metrics["test_PER"]["precision"] * metrics["test_PER"]["number"]
            + metrics["test_ORG"]["precision"] * metrics["test_ORG"]["number"]
            + metrics["test_LOC"]["precision"] * metrics["test_LOC"]["number"]
            + metrics["test_ANIM"]["precision"] * metrics["test_ANIM"]["number"]
            + metrics["test_DIS"]["precision"] * metrics["test_DIS"]["number"]
    ) / total_support

    metrics["test_overall_f1"] = 2 * metrics["test_overall_precision"] * metrics["test_overall_recall"] \
                                 / (metrics["test_overall_precision"] + metrics["test_overall_recall"])

    if log_to_wandb:
        wandb.log({"test/overall_precision": metrics["test_overall_precision"]})
        wandb.log({"test/overall_f1": metrics["test_overall_f1"]})

    return metrics
