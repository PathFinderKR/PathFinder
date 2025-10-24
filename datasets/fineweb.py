import os
from transformers import AutoTokenizer
from datasets import load_dataset
from src.config import DatasetConfig, TokenizerConfig
NUM_PROC = 4


def preprocess_batch(batch, tokenizer, max_seq_len=1024):
    """
    Preprocess a batch of text data by tokenizing and adding end-of-text tokens.

    Args:
        batch: A batch of data containing text.
        tokenizer: The tokenizer to use for encoding.
        max_seq_len: The maximum sequence length for truncation.

    Returns:
        A dictionary containing the tokenized input IDs.
    """
    eot_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    encoded_batch = tokenizer.batch_encode_plus(
        batch["text"],
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max_seq_len - 1,
        return_attention_mask=False
    )
    input_ids = [[eot_id] + input_ids for input_ids in encoded_batch["input_ids"]]

    return {"input_ids": input_ids}


def main():
    # Configuration
    dataset_config = DatasetConfig()
    tokenizer_config = TokenizerConfig()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_id,
        use_fast=True
    )
    print(f"Tokenizer: {tokenizer}")

    # Load FineWeb-Edu dataset
    fineweb = load_dataset(
        dataset_config.dataset_id,
        name=dataset_config.remote_name,
        split=dataset_config.split,
        num_proc=NUM_PROC
    )
    print(f"FineWeb-Edu dataset: {fineweb}")

    # Tokenize the dataset
    fineweb_tokenized = fineweb.map(
        lambda batch: preprocess_batch(batch=batch, tokenizer=tokenizer),
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["text"]
    )

    # Save the tokenized dataset
    os.makedirs(dataset_config.local_dir, exist_ok=True)
    fineweb_tokenized.save_to_disk(dataset_config.local_dir)
    print(f"Tokenized dataset saved to {dataset_config.local_dir}")


if __name__ == "__main__":
    main()