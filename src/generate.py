import os
import torch
from transformers import AutoTokenizer
from src.utils import set_seed
from models.GPT import GPT
from config import TokenizerConfig, ModelConfig, GenerationConfig


def main():
    # Configuration
    tokenizer_config = TokenizerConfig()
    model_config = ModelConfig()
    generation_config = GenerationConfig()

    # Device
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    torch.set_float32_matmul_precision(generation_config.matmul_precision)

    # Reproducibility
    set_seed(42)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_id,
        use_fast=True
    )
    print(f"Tokenizer: {tokenizer}")

    # Model
    model = GPT(model_config)
    if os.path.exists(generation_config.checkpoint_path):
        model = model.from_pretrained(generation_config.checkpoint_path).to(device)
    else:
        raise FileNotFoundError(f"No checkpoint found at {generation_config.checkpoint_path}")
    print(model)

    # Generate
    while True:
        print("=" * 50)
        print("User prompt: ")
        user_prompt = input("> ")
        if user_prompt.lower() == "exit":
            break
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
        print("-" * 50)
        print("🤖 Model Response:")
        output = model.generate(
            input_ids,
            use_cache=generation_config.use_cache,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            tokenizer=tokenizer
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)


if __name__ == "__main__":
    main()