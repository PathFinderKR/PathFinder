import os
from transformers import AutoTokenizer
from datasets import load_dataset
from src.config import DatasetConfig, TokenizerConfig


def preprocess_batch(batch, tokenizer, eot_id, max_seq_len=1024):
    """
    Preprocess a batch of text data by tokenizing and adding end-of-text tokens.


    Args:
        batch: A batch of data containing text.
        tokenizer: The tokenizer to use for encoding.
        eot_id: The ID of the end-of-text token.
        max_seq_len: The maximum sequence length for truncation.

    Returns:
        A dictionary containing the tokenized input IDs.
    """
    texts = batch["text"]
    input_ids_batch = []

    for text in texts:
        input_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len - 1,
        )
        input_ids = [eot_id] + input_ids
        input_ids_batch.append(input_ids)

    return {"input_ids": input_ids_batch}


def main():
    dataset_config = DatasetConfig()
    tokenizer_config = TokenizerConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_id,
        use_fast=True
    )
    print(f"Tokenizer: {tokenizer}")
    eot_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    num_proc = max(1, os.cpu_count() // 2)
    print(f"Number of processors: {num_proc}")

    fineweb = load_dataset(dataset_config.dataset_id, name=dataset_config.remote_name, split=dataset_config.split, num_proc=num_proc)
    print(f"FineWeb-Edu dataset: {fineweb}")

    fineweb_tokenized = fineweb.map(
        lambda batch: preprocess_batch(batch=batch, tokenizer=tokenizer, eot_id=eot_id),
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"]
    )

    os.makedirs(dataset_config.local_dir, exist_ok=True)
    fineweb_tokenized.save_to_disk(dataset_config.local_dir)


if __name__ == "__main__":
    main()