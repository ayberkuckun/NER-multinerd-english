from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification


def get_model(configs, system):
    """
    Gets the model either from HuggingFace or disk.
    """
    model = AutoModelForTokenClassification.from_pretrained(
        configs[system]["model_checkpoint"], id2label=configs[system]["id2label"], label2id=configs[system]["label2id"])

    return model


def get_tokenizer(configs, system):
    """
    Gets the tokenizer from HuggingFace.
    """
    tokenizer = AutoTokenizer.from_pretrained(configs[system]["model_checkpoint"], use_fast=True)

    return tokenizer


def get_data_collator(tokenizer):
    """
    Gets the Data Collator.
    """
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    return data_collator
