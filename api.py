from flask import Flask, request, jsonify
import csv
import os
import random
from datetime import datetime
from utils import load_model, clean_text
from evaluate import generate_feedback
from config import DEADLINE

app = Flask(__name__)
model, tokenizer = load_model()

@app.route("/submit_proposal", methods=["POST"])
def submit_proposal():
    current_time = datetime.now()

    if current_time > DEADLINE:
        return jsonify({"error": "Submission deadline has passed. No more proposals accepted."}), 403

    data = request.json
    user_id = data.get("user_id")
    proposal = clean_text(data.get("proposal", ""))

    if not user_id or not proposal:
        return jsonify({"error": "User ID and proposal are required"}), 400

    csv_file = "submissions.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", "proposal", "group", "feedback"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "user_id": user_id,
            "proposal": proposal,
            "group": "",       
            "feedback": ""
        })

    return jsonify({"message": "Proposal submitted successfully.", "user_id": user_id})

