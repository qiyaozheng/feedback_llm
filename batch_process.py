import csv
import os
import random
from utils import load_model, clean_text
from evaluate import generate_feedback

model, tokenizer = load_model()

input_csv = "submissions.csv"
output_csv = "evaluated_feedback.csv"

def batch_process():
    if not os.path.exists(input_csv):
        print("No proposals found for evaluation.")
        return

    results = []
    with open(input_csv, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = row["user_id"]
            proposal = clean_text(row['Proposal'])

            group = random.choice(["Improvement Feedback", "Redundancy Feedback"])
            feedback = generate_feedback(user_id, proposal, group, model, tokenizer)

            results.append({
                "user_id": user_id,
                "proposal": proposal,
                "group": group,
                "feedback": feedback
            })

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "proposal", "group", "feedback"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Batch processing complete! Results saved to {output_csv}.")

if __name__ == "__main__":
    batch_process()