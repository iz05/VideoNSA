import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import fire
import lmms_eval.models

def av_asd_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["prompt"]

def av_asd_doc_to_visual(doc):
    return doc["video"]

def av_asd_process_results(doc, results):
    # Example results string: "0,0,1,1,0,0,"
    # Remove whitespace and trailing commas
    raw = results[0].strip().rstrip(",")

    # Split into tokens
    if raw:
        try:
            predicted = [int(x) for x in raw.split(",")]
        except Exception as e:
            print(f"Uh oh, got exception {e}")
            predicted = []
    else:
        predicted = []

    correct = doc["correct_answers"]

    # Ensure same length (truncate to the shorter length)
    L = min(len(predicted), len(correct))
    predicted = predicted[:L]
    correct = correct[:L]

    # Compute correctness
    correct_detailed = [1 if p == c else 0 for p, c in zip(predicted, correct)]
    all_correct = int(all(correct_detailed))

    return {
        "results": {
            "video_id": doc["uid"],
            "correct": all_correct,          # 1 if all match, else 0
            "correct_detailed": correct_detailed,
            "predicted_answer": predicted,
            "correct_answer": correct,
        }
    }