import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)
else:
    raise ValueError("❌ Hugging Face Token not found. Please set HF_TOKEN in .env.")

def load_and_save_model(model_name="meta-llama/Llama-3.1-8B-Instruct", save_path="models/trained_model"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True).to(device)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"🎉 Model and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    load_and_save_model()
