import datasets

from llama_recipes.datasets.utils import Concatenator
import pandas as pd
import modal
import re



B_INST, E_INST = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def format_text(row, tokenizer):
    text = (
        B_INST
        + B_SYS
        + "You are a documentation assistant that finds the topic of a user question that is then used to search the documentation.\n"
        + "Here is an example:\nQuestion: How do I upload a file to a volume?\nAnswer: modal volume modal volume put\n"
        + E_SYS
        + row["question"]
        + E_INST
        + "\n"
        + row["topic"]
        + "\n"
        + "</s>"
    )

    return tokenizer(text)


def get_custom_dataset(dataset_config, tokenizer, split):
    full_dataset = datasets.load_dataset('csv', data_files = 'dataset.csv', split="train")
    print(f"TYPE: {type(full_dataset)}")


    # Since the dataset has no train/test split, we create one and select it
    dataset = full_dataset.train_test_split(
        train_size=0.8,
        seed=42,
    )["train" if split == dataset_config.train_split else "test"]

    dataset = dataset.map(
        lambda x: format_text(x, tokenizer), remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True, batch_size=None)

    return dataset
