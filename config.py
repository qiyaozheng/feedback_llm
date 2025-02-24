from datetime import datetime
DEADLINE = datetime.strptime("2025-02-25 12:00:00", "%Y-%m-%d %H:%M:%S") #example
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAVE_PATH = "models/trained_model"
INPUT_PATH = "data/input.xlsx"
OUTPUT_PATH = "output/feedback_results.csv"