import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import fire
import lmms_eval.models

# -----------------------------
# HourVideo helper functions
# -----------------------------
def hourvideo_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    choices = doc.get("mcq_test", "")

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
        question = f"{pre_prompt}{question}{post_prompt}"

    prompt = (
        f"Select the best answer to the following multiple-choice question based on the video.\n"
        f"{question}\n{choices}\nRespond with only the letter (A, B, C, D, or E)."
    )
    return prompt

def hourvideo_doc_to_visual(doc):
    return doc["video"]  # list of PIL frames

def hourvideo_process_results(doc, results):
    pred = results[0].lower().strip()
    predicted = None
    for c in pred:
        if c.isalpha():
            predicted = c.upper()
            break
    if predicted is None:
        predicted = "Z"

    correct = predicted.upper() == doc["correct_answer_label"].upper()
    return {
        "hv_acc": {
            "question_id": doc["uid"],
            "score": 1.0 if correct else 0.0,
            "predicted_answer": predicted,
            "correct_answer": doc["correct_answer_label"].upper(),
        }
    }

def hourvideo_aggregate_results(results):
    total, correct = 0, 0
    for r in results:
        for val in r.values():
            total += 1
            correct += val["score"]
    acc = correct / total if total > 0 else 0.0
    return acc