import datasets


def get_dataset(tokenizer, configs, system, load_ds_from_disk=False, preprocess=True):
    """
    Loads the dataset of the system either from disk or HuggingFace and preprocesses it if desired.
    """
    if load_ds_from_disk:
        dataset = datasets.load_from_disk(configs[system]["dataset_path"])
    else:
        dataset = datasets.load_dataset(configs["dataset_name"], num_proc=configs["num_proc"])

    if preprocess:
        dataset = _preprocess_dataset(tokenizer, dataset, configs, system)

    return dataset


def _preprocess_dataset(tokenizer, dataset, configs, system):
    """
    Preprocesses the dataset. The process consist of extracting the examples of English language, removing unnecessary
    columns, tokenizing and aligning the labels. Saves the processed dataset to disk for further use.

    According to the defined system type, applies the required changes to dataset.
    """
    dataset = dataset.filter(lambda sentence: sentence["lang"] == "en", num_proc=configs["num_proc"])
    dataset = dataset.select_columns(["tokens", "ner_tags"])

    if system == "systemB":
        dataset = dataset.map(_get_process_system_b_dataset_fn(configs), num_proc=configs["num_proc"])

    tokenized_dataset = dataset.map(
        _get_tokenize_and_align_dataset_fn(tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names,
        num_proc=configs["num_proc"]
    )
    tokenized_dataset.save_to_disk(configs[system]['dataset_path'])

    return tokenized_dataset


def _get_process_system_b_dataset_fn(configs):
    """
    Getter of process_system_b_dataset function.
    """
    def process_system_b_dataset(example):
        """
        Processes dataset in order to remove entity tags specified by the user.
        Fixes the numbering of the tags as specified by the user.
        """
        for i, tag in enumerate(example["ner_tags"]):
            if tag in configs["systemB"]["removed_tags"]:
                example["ner_tags"][i] = 0

            elif tag in configs["systemB"]["changed_tags"].keys():
                example["ner_tags"][i] = configs["systemB"]["changed_tags"][tag]

        return example

    return process_system_b_dataset


def _get_tokenize_and_align_dataset_fn(tokenizer):
    """
    Getter of tokenize_and_align_dataset function.
    """
    def tokenize_and_align_dataset(example):
        """
        Tokenizes and aligns the dataset. Can work with batches.
        """
        aligned_labels = []
        labels = example["ner_tags"]
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

        for i, label in enumerate(labels):
            aligned_label = _align_labels_with_tokens(label, tokenized_inputs.word_ids(i))
            aligned_labels.append(aligned_label)

        tokenized_inputs["labels"] = aligned_labels

        return tokenized_inputs

    return tokenize_and_align_dataset


def _align_labels_with_tokens(labels, token_word_ids):
    """
    Takes true labels and tokens of a single example as input. Returns aligned labels.
    """
    aligned_labels = []
    pre_token_word_id = None
    for token_word_id in token_word_ids:
        if token_word_id is None:
            aligned_labels.append(-100)

        elif token_word_id == pre_token_word_id:
            # Check if we need to change the entity tag from "B-" to "I-"
            if labels[token_word_id] % 2 == 1:
                aligned_labels.append(labels[token_word_id] + 1)

            else:
                aligned_labels.append(labels[token_word_id])

        else:
            aligned_labels.append(labels[token_word_id])

        pre_token_word_id = token_word_id

    return aligned_labels


def get_prepare_system_a_test_set_fn(configs):
    """
    Getter of prepare_system_a_test_set function.
    """
    def prepare_system_a_test_set(example):
        """
        Fixes the tag numbering as suitable to the system A training dataset.
        """
        reverse_mapping = {v: k for k, v in configs["systemB"]["changed_tags"].items()}

        for i, tag in enumerate(example["labels"]):
            if tag in reverse_mapping.keys():
                example["labels"][i] = reverse_mapping[tag]

        return example

    return prepare_system_a_test_set

