import os
import torch
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import set_seed
from models.GPT2 import GPT
from config import TokenizerConfig, ModelConfig, GenerationConfig


def main():
    # Reproducibility
    set_seed(42)

    # Configuration
    tokenizer_config = TokenizerConfig()
    model_config = ModelConfig()
    generation_config = GenerationConfig()

    # Device
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_id)

    # Model
    model = GPT(model_config).to(device=device)
    model = model.compile(model)
    if os.path.exists(generation_config.checkpoint_path):
        checkpoint = torch.load(generation_config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"No checkpoint found at {generation_config.checkpoint_path}")

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
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)


if __name__ == "__main__":
    main()