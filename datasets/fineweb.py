import os
from typing import Literal, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer
from datasets import load_dataset

@dataclass
class Config:
    tokenizer_id: Literal["gpt2"] = "gpt2"
    max_seq_len: int = 128
    dataset_id: Literal["HuggingFaceFW/fineweb-edu"] = "HuggingFaceFW/fineweb-edu"
    remote_name: Optional[str] = "sample-10BT"
    split: Optional[str] = "train"
    local_dir: str = f"FineWeb-Edu/10B-{max_seq_len}"


def preprocess_batch(batch, tokenizer, eot_id, max_seq_len):
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
    config = Config()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    eot_token = tokenizer.eos_token
    eot_id = tokenizer.convert_tokens_to_ids(eot_token)

    num_proc = max(1, os.cpu_count() // 2)
    print(f"Number of processors: {num_proc}")

    fineweb = load_dataset(config.dataset_id, name=config.remote_name, split=config.split, num_proc=num_proc)
    print(f"FineWeb-Edu dataset: {fineweb}")

    fineweb_tokenized = fineweb.map(
        lambda batch: preprocess_batch(batch, tokenizer, eot_id, config.max_seq_len),
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"]
    )

    fineweb_tokenized.save_to_disk(config.local_dir)


if __name__ == "__main__":
    main()