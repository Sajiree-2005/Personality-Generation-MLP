import os
import logging
import time
from pathlib import Path

# Keep TensorFlow from creating large CPU thread pools on small hosting instances.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = OUTPUT_DIR / "personality_mlp_model.keras"
SCALER_PATH = OUTPUT_DIR / "personality_input_scaler.pkl"
METADATA_PATH = OUTPUT_DIR / "personality_model_metadata.pkl"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load artifacts once when the web process starts, not on every request.
_start = time.perf_counter()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
metadata = joblib.load(METADATA_PATH)
app.logger.info("Model artifacts loaded in %.2f seconds", time.perf_counter() - _start)

QUESTION_COLUMNS = metadata["question_columns"]
TRAITS = metadata["traits"]
TRAIT_DESCRIPTIONS = metadata["trait_descriptions"]

QUESTION_TEXT = {
    "EXT1": "I am the life of the party.",
    "EXT2": "I don't talk a lot.",
    "EXT3": "I feel comfortable around people.",
    "EXT4": "I keep in the background.",
    "EXT5": "I start conversations.",
    "EXT6": "I have little to say.",
    "EXT7": "I talk to a lot of different people at parties.",
    "EXT8": "I don't like to draw attention to myself.",
    "EXT9": "I don't mind being the center of attention.",
    "EXT10": "I am quiet around strangers.",
    "EST1": "I get stressed out easily.",
    "EST2": "I am relaxed most of the time.",
    "EST3": "I worry about things.",
    "EST4": "I seldom feel blue.",
    "EST5": "I am easily disturbed.",
    "EST6": "I get upset easily.",
    "EST7": "I change my mood a lot.",
    "EST8": "I have frequent mood swings.",
    "EST9": "I get irritated easily.",
    "EST10": "I often feel blue.",
    "AGR1": "I feel little concern for others.",
    "AGR2": "I am interested in people.",
    "AGR3": "I insult people.",
    "AGR4": "I sympathize with others' feelings.",
    "AGR5": "I am not interested in other people's problems.",
    "AGR6": "I have a soft heart.",
    "AGR7": "I am not really interested in others.",
    "AGR8": "I take time out for others.",
    "AGR9": "I feel others' emotions.",
    "AGR10": "I make people feel at ease.",
    "CSN1": "I am always prepared.",
    "CSN2": "I leave my belongings around.",
    "CSN3": "I pay attention to details.",
    "CSN4": "I make a mess of things.",
    "CSN5": "I get chores done right away.",
    "CSN6": "I often forget to put things back in their proper place.",
    "CSN7": "I like order.",
    "CSN8": "I shirk my duties.",
    "CSN9": "I follow a schedule.",
    "CSN10": "I am exacting in my work.",
    "OPN1": "I have a rich vocabulary.",
    "OPN2": "I have difficulty understanding abstract ideas.",
    "OPN3": "I have a vivid imagination.",
    "OPN4": "I am not interested in abstract ideas.",
    "OPN5": "I have excellent ideas.",
    "OPN6": "I do not have a good imagination.",
    "OPN7": "I am quick to understand things.",
    "OPN8": "I use difficult words.",
    "OPN9": "I spend time reflecting on things.",
    "OPN10": "I am full of ideas.",
}

GROUPS = [
    ("Extraversion", "EXT", "How you engage with people and social situations"),
    ("Emotional responses", "EST", "Items used by this project to estimate Neuroticism"),
    ("Agreeableness", "AGR", "How you relate to and consider other people"),
    ("Conscientiousness", "CSN", "How you approach planning, order, and responsibilities"),
    ("Openness", "OPN", "Your interest in ideas, imagination, and new experiences"),
]

def trait_level(score):
    if score < 2.5:
        return "Low"
    if score < 3.5:
        return "Medium"
    return "High"

def predict(answers):
    # Preserve the exact feature order used to train the model.
    row = pd.DataFrame([[answers[col] for col in QUESTION_COLUMNS]], columns=QUESTION_COLUMNS)
    _start = time.perf_counter()
    scaled = scaler.transform(row).astype(np.float32, copy=False)
    app.logger.info("Prediction preprocessing completed in %.3f seconds", time.perf_counter() - _start)

    # Calling the model directly avoids Keras predict() data-adapter overhead for one row.
    _start = time.perf_counter()
    raw_scores = np.asarray(model(scaled, training=False))[0]
    app.logger.info("Model inference completed in %.3f seconds", time.perf_counter() - _start)
    scores = np.clip(raw_scores, 1, 5)
    results = []
    score_map = {}
    for trait, score in zip(TRAITS, scores):
        score = float(score)
        level = trait_level(score)
        score_map[trait] = score
        results.append({
            "name": trait,
            "score": round(score, 2),
            "level": level,
            "percent": round(score / 5 * 100),
            "description": TRAIT_DESCRIPTIONS[trait][level],
        })
    summary = " ".join(
        f"{item['name']} is estimated as {item['level'].lower()} "
        f"({item['score']:.2f}/5), suggesting that you {item['description']}."
        for item in results
    )
    return results, summary

@app.get("/")
def index():
    groups = []
    for title, prefix, subtitle in GROUPS:
        questions = [
            {"code": code, "text": QUESTION_TEXT[code]}
            for code in QUESTION_COLUMNS if code.startswith(prefix)
        ]
        groups.append({"title": title, "prefix": prefix, "subtitle": subtitle, "questions": questions})
    return render_template("index.html", groups=groups, results=None, summary=None, answers={})

@app.post("/predict")
def predict_route():
    _request_start = time.perf_counter()
    app.logger.info("Received POST /predict")
    answers = {}
    errors = []
    for code in QUESTION_COLUMNS:
        value = request.form.get(code, "").strip()
        if value not in {"1", "2", "3", "4", "5"}:
            errors.append(code)
        else:
            answers[code] = int(value)

    groups = []
    for title, prefix, subtitle in GROUPS:
        questions = [
            {"code": code, "text": QUESTION_TEXT[code]}
            for code in QUESTION_COLUMNS if code.startswith(prefix)
        ]
        groups.append({"title": title, "prefix": prefix, "subtitle": subtitle, "questions": questions})

    if errors:
        return render_template(
            "index.html", groups=groups, results=None, summary=None, answers=answers,
            error=f"Please answer all 50 statements. {len(errors)} answer(s) are missing or invalid."
        ), 400

    try:
        results, summary = predict(answers)
    except Exception:
        app.logger.exception("Prediction failed while processing POST /predict")
        return "Prediction failed on the server. Check the service logs for the detailed error.", 500

    app.logger.info("POST /predict completed in %.3f seconds", time.perf_counter() - _request_start)
    return render_template(
        "index.html", groups=groups, results=results, summary=summary, answers=answers, error=None
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
