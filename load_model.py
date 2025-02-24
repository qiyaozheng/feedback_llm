import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_and_save_model(model_name="meta-llama/Llama-3.1-8B-Instruct", save_path="models/trained_model"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    load_and_save_model()
