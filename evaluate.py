import pandas as pd
import random
from utils import load_model

model, tokenizer = load_model()

def generate_feedback(user_id, proposal, group):

    improvement_few_shot = """

--- Few-shot Example 1: Improvement Feedback ---
Proposal: #TODO
Improvement Feedback: #TODO

--- Few-shot Example 2: Improvement Feedback ---
Proposal: #TODO
Improvement Feedback: #TODO

"""

    redundancy_few_shot = """

--- Few-shot Example 1: Redundancy  Feedback ---
Proposal: #TODO
Improvement Feedback: #TODO

--- Few-shot Example 2: Redundancy Feedback ---
Proposal: #TODO
Improvement Feedback: #TODO

"""

    if group == "Improvement Feedback":
        prompt = f"""
You are an expert evaluator for hackathon proposals. Your task is to provide feedback based on the proposal.

{improvement_few_shot}

--- New Proposal to Evaluate ---
Proposal: {proposal}

Please provide 3-4 concrete suggestions for improving the following solution. Each suggestion should be presented as a separate paragraph. Focus on the following aspects:

1. Identify existing weaknesses and explain why improvements are needed.
2. Propose clear, actionable steps to enhance effectiveness and innovation.
"""
    else:
        prompt = f"""
You are an expert evaluator for hackathon proposals. Your task is to provide feedback based on the proposal.

{redundancy_few_shot}

--- New Proposal to Evaluate ---
Proposal: {proposal}

Please identify 3-4 aspects of the following solution that are either redundant, highly similar to existing solutions, or uniquely innovative. Present the output as follows:

1. **1-2 aspects that are not novel:** Highlight features that resemble existing solutions and explain why they lack uniqueness.
2. **1-2 aspects that are novel and uncommon:** Identify distinctive features worth retaining and elaborate on their value.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs["input_ids"], max_length=512, temperature=0.7)
    feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"user_id": user_id, "proposal": proposal, "group": group, "feedback": feedback}