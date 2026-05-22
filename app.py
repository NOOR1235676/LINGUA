"""
Lingua — N-gram Language Model Web App
FastAPI backend serving auto-complete, sentiment, and perplexity endpoints.

Run with:  python app.py
Then open: http://localhost:8000
"""

import math
import pickle
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# nltk for tokenization
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
STATIC_DIR = BASE_DIR / "static"


# ============================================================
# LOAD MODELS FROM PICKLE
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"model.pkl not found at {MODEL_PATH}.\n"
        f"Run the export snippet from your Colab notebook first (see README.md)."
    )

print(f"Loading model from {MODEL_PATH} ...")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

n_gram_counts_list = data["n_gram_counts_list"]
vocabulary         = data["vocabulary"]
unigram_counts     = n_gram_counts_list[0]
bigram_counts      = n_gram_counts_list[1]
trigram_counts     = n_gram_counts_list[2]

pos_model_v2       = data["pos_model_v2"]
neg_model_v2       = data["neg_model_v2"]
pos_model_base     = data.get("pos_model")
neg_model_base     = data.get("neg_model")

NEGATION_WORDS     = set(data["NEGATION_WORDS"])
PUNCT_BOUNDARY     = set(data["PUNCT_BOUNDARY"])
POSITIVE_WORDS     = set(data["POSITIVE_WORDS"])
NEGATIVE_WORDS     = set(data["NEGATIVE_WORDS"])

print(f"  vocabulary size  : {len(vocabulary):,}")
print(f"  n-gram orders    : 1 .. {len(n_gram_counts_list)}")
print(f"  pos model vocab  : {len(pos_model_v2['vocab']):,}")
print(f"  neg model vocab  : {len(neg_model_v2['vocab']):,}")
print("Model loaded.")


# ============================================================
# INFERENCE FUNCTIONS
# (mirror your Colab functions exactly)
# ============================================================

def estimate_probability(word, previous_n_gram, n_gram_counts,
                         n_plus1_gram_counts, vocabulary_size, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    prev_count = n_gram_counts.get(previous_n_gram, 0)
    denom = prev_count + k * vocabulary_size
    n_plus1 = previous_n_gram + (word,)
    num = n_plus1_gram_counts.get(n_plus1, 0) + k
    return num / denom


def estimate_probabilities(previous_n_gram, n_gram_counts,
                           n_plus1_gram_counts, vocabulary, k=1.0):
    previous_n_gram = tuple(previous_n_gram)
    vocab = vocabulary + ["<e>"]
    vocab_size = len(vocab)
    probs = {}
    for word in vocab:
        probs[word] = estimate_probability(
            word, previous_n_gram, n_gram_counts,
            n_plus1_gram_counts, vocab_size, k
        )
    return probs


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts,
                   vocabulary, k=1.0, start_with=None, start_token="<s>"):
    n = len(next(iter(n_gram_counts)))
    if len(previous_tokens) < n:
        previous_tokens = [start_token] * (n - len(previous_tokens)) + list(previous_tokens)
    vocab_set = set(vocabulary)
    previous_tokens = [t if t in vocab_set else "<unk>" for t in previous_tokens]
    previous_n_gram = tuple(previous_tokens[-n:])
    probabilities = estimate_probabilities(
        previous_n_gram, n_gram_counts, n_plus1_gram_counts,
        vocabulary, k=k
    )
    suggestion, max_prob = None, 0.0
    for word, prob in probabilities.items():
        if word in ("<unk>", "<s>", "<e>"):
            continue
        if start_with is not None and not word.startswith(start_with):
            continue
        if prob > max_prob:
            suggestion, max_prob = word, prob
    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list_in, vocabulary,
                    k=1.0, start_with=None):
    suggestions = []
    for i in range(len(n_gram_counts_list_in) - 1):
        sug = suggest_a_word(
            previous_tokens,
            n_gram_counts_list_in[i],
            n_gram_counts_list_in[i + 1],
            vocabulary, k=k, start_with=start_with
        )
        suggestions.append(sug)
    return suggestions


def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts,
                         vocabulary_size, k=1.0):
    n = len(next(iter(n_gram_counts)))
    sentence = ["<s>"] * n + list(sentence) + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    log_prob_sum = 0.0
    num_predictions = N - n
    for t in range(n, N):
        prev = sentence[t - n:t]
        word = sentence[t]
        prob = estimate_probability(
            word, prev, n_gram_counts, n_plus1_gram_counts,
            vocabulary_size, k
        )
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / num_predictions)


def apply_negation_marking(tokens, scope_length=2):
    result = []
    remaining = 0
    for tok in tokens:
        if tok in PUNCT_BOUNDARY:
            remaining = 0
            result.append(tok)
        elif tok in NEGATION_WORDS:
            remaining = scope_length
            result.append(tok)
        elif remaining > 0:
            result.append("NOT_" + tok)
            remaining -= 1
        else:
            result.append(tok)
    return result


def lexicon_sentiment_score(tokens, scope=2):
    score = 0
    negate_remaining = 0
    for tok in tokens:
        if tok in PUNCT_BOUNDARY:
            negate_remaining = 0
            continue
        if tok in NEGATION_WORDS:
            negate_remaining = scope
            continue
        if tok in POSITIVE_WORDS:
            score += -1 if negate_remaining > 0 else +1
            if negate_remaining > 0:
                negate_remaining -= 1
        elif tok in NEGATIVE_WORDS:
            score += +1 if negate_remaining > 0 else -1
            if negate_remaining > 0:
                negate_remaining -= 1
    return score


def classify_sentiment_v3(sentence, k=1.0):
    """
    Hybrid sentiment classifier with four decision paths:
      Rule 1: negation present       -> trust lexicon
      Rule 2: out-of-domain input    -> trust lexicon (huge perplexity gap or both high)
      Rule 3: bigram low confidence  -> trust lexicon (NEW)
      Otherwise: trust bigram model
    """
    tokens = nltk.word_tokenize(sentence.lower())
    has_negation = any(t in NEGATION_WORDS for t in tokens)
    lex_score = lexicon_sentiment_score(tokens)
    lex_pred = "pos" if lex_score > 0 else ("neg" if lex_score < 0 else None)

    # Bigram models with negation marking
    tokens_neg = apply_negation_marking(tokens)
    pos_proc = [t if t in pos_model_v2["vocab_set"] else "<unk>" for t in tokens_neg]
    neg_proc = [t if t in neg_model_v2["vocab_set"] else "<unk>" for t in tokens_neg]

    pp_pos = calculate_perplexity(
        pos_proc, pos_model_v2["unigram"], pos_model_v2["bigram"],
        len(pos_model_v2["vocab"]), k=k
    )
    pp_neg = calculate_perplexity(
        neg_proc, neg_model_v2["unigram"], neg_model_v2["bigram"],
        len(neg_model_v2["vocab"]), k=k
    )
    bigram_pred = "pos" if pp_pos < pp_neg else "neg"

    # Override signals
    perp_ratio = max(pp_pos, pp_neg) / max(min(pp_pos, pp_neg), 1)
    is_ood = perp_ratio > 5 or min(pp_pos, pp_neg) > 500

    bigram_margin = abs(pp_pos - pp_neg) / max(pp_pos, pp_neg)
    bigram_uncertain = bigram_margin < 0.20   # less than 20% margin = uncertain

    # Decision tree (order matters)
    if has_negation and lex_pred:
        label, source = lex_pred, "lexicon (negation)"
    elif is_ood and lex_pred:
        label, source = lex_pred, "lexicon (out-of-domain)"
    elif bigram_uncertain and lex_pred:
        label, source = lex_pred, "lexicon (low confidence)"
    else:
        label, source = bigram_pred, "bigram"

    return label, pp_pos, pp_neg, lex_score, source


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Lingua — N-gram Language Model",
    description="Auto-complete, sentiment, and perplexity API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class AutocompleteRequest(BaseModel):
    text: str = ""
    prefix: str = ""


class SentimentRequest(BaseModel):
    text: str


class PerplexityRequest(BaseModel):
    sentence1: str
    sentence2: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/api/info")
def api_info():
    return {
        "vocabulary_size": len(vocabulary),
        "n_gram_orders": len(n_gram_counts_list),
        "model": "n-gram with k=1 Laplace smoothing",
        "version": "1.0.0",
    }


@app.post("/api/autocomplete")
def api_autocomplete(req: AutocompleteRequest):
    raw = req.text.lower()
    manual_prefix = req.prefix.strip().lower() if req.prefix else ""

    if not raw.strip() and not manual_prefix:
        return {
            "context_tokens": [],
            "filter_prefix": None,
            "suggestions": [],
        }

    if not raw.endswith(" "):
        parts = raw.strip().split()
        auto_prefix = parts[-1] if parts else None
        clean_text = " ".join(parts[:-1])
    else:
        auto_prefix = None
        clean_text = raw.strip()

    final_prefix = manual_prefix or auto_prefix
    tokens = nltk.word_tokenize(clean_text) if clean_text else []
    suggestions = get_suggestions(
        tokens, n_gram_counts_list, vocabulary,
        k=1.0, start_with=final_prefix
    )

    return {
        "context_tokens": tokens,
        "filter_prefix": final_prefix,
        "suggestions": [
            {
                "order": i + 2,
                "word": word,
                "probability": prob,
            }
            for i, (word, prob) in enumerate(suggestions)
        ],
    }


@app.post("/api/sentiment")
def api_sentiment(req: SentimentRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    label, pp_pos, pp_neg, lex_score, source = classify_sentiment_v3(req.text)
    margin = abs(pp_pos - pp_neg) / max(pp_pos, pp_neg)

    return {
        "label": label,
        "pp_pos": pp_pos,
        "pp_neg": pp_neg,
        "lex_score": lex_score,
        "source": source,
        "margin": margin,
        "text": req.text,
    }


@app.post("/api/perplexity")
def api_perplexity(req: PerplexityRequest):
    if not req.sentence1.strip() or not req.sentence2.strip():
        raise HTTPException(status_code=400, detail="Both sentences required.")

    vset = set(vocabulary)
    t1 = [t if t in vset else "<unk>" for t in nltk.word_tokenize(req.sentence1.lower())]
    t2 = [t if t in vset else "<unk>" for t in nltk.word_tokenize(req.sentence2.lower())]

    pp1 = calculate_perplexity(t1, unigram_counts, bigram_counts, len(vocabulary), k=1.0)
    pp2 = calculate_perplexity(t2, unigram_counts, bigram_counts, len(vocabulary), k=1.0)

    return {
        "sentence1": {"text": req.sentence1, "perplexity": pp1},
        "sentence2": {"text": req.sentence2, "perplexity": pp2},
        "winner": "sentence1" if pp1 < pp2 else "sentence2",
    }


# ============================================================
# STATIC FILES (serves the HTML frontend)
# ============================================================

if STATIC_DIR.exists():
    @app.get("/")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print(" Lingua server starting...")
    print(" Open http://localhost:8000 in your browser")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")