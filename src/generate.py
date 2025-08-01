import os
import sys
import torch
from transformers import AutoTokenizer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.utils import set_seed, speedometer
from models.GPT import GPT
from src.config import TokenizerConfig, GenerationConfig

def main():
    # Configuration
    tokenizer_config = TokenizerConfig()
    generation_config = GenerationConfig()

    # Device
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    torch.set_float32_matmul_precision(generation_config.matmul_precision)
    print(f"MatMul Precision: {generation_config.matmul_precision}")

    # Reproducibility
    set_seed(generation_config.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config.tokenizer_id,
        use_fast=True
    )
    print(f"Tokenizer: {tokenizer}")

    # Model
    if generation_config.model_id is not None:
        # Download from hub
        from huggingface_hub import snapshot_download
        local_dir = os.path.join(PROJECT_ROOT, "checkpoints", generation_config.model_id)
        snapshot_download(
            repo_id=f"PathFinderKR/{generation_config.model_id}",
            local_dir=local_dir,
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        model = GPT.from_pretrained(local_dir).to(device)
    else:
        # Load from local directory
        checkpoint_path = os.path.join(PROJECT_ROOT, generation_config.checkpoint_path)
        if os.path.exists(checkpoint_path):
            model = GPT.from_pretrained(checkpoint_path).to(device)
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model = torch.compile(model)
    print(model)
    print(f"Number of parameters: {model.num_params() / 1e6:.2f}M")

    # Speedometer
    if generation_config.speedometer:
        speedometer(
            model=model,
            input_ids=tokenizer.encode("a", return_tensors="pt").to(device),
            use_cache=False
        )
        speedometer(
            model=model,
            input_ids=tokenizer.encode("a", return_tensors="pt").to(device),
            use_cache=True
        )

    # Generate
    while True:
        print("=" * 60)
        print("User prompt: ")
        user_prompt = input("> ")
        if user_prompt.lower() == "exit":
            break
        input_ids = tokenizer.encode(user_prompt, return_tensors="pt").to(device)
        print("-" * 60)
        print("🤖 Model Response:")
        output = model.generate(
            input_ids,
            use_cache=True,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            tokenizer=tokenizer
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(response)


if __name__ == "__main__":
    main()