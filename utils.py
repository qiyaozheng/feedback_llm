import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def clean_text(text):
    text = re.sub(r'[^\w\s.]', '', text)
    return text.strip()

def load_model(model_path="models/trained_model"):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    return model, tokenizer 