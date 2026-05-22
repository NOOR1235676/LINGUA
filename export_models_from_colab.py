"""
RUN THIS IN YOUR COLAB NOTEBOOK to export trained models for the local web app.

Add this as a new cell at the END of your notebook (after all your models are built).
Run it once. It will produce a 'model.pkl' file in your Colab session — download it,
then drop it into the 'lingua/' folder alongside app.py.
"""

# ====== Paste this code as a new cell in your Colab notebook ======

import pickle
from google.colab import files  # for download in Colab

# Bundle everything the backend needs
model_bundle = {
    "n_gram_counts_list": n_gram_counts_list,
    "vocabulary": vocabulary,
    "pos_model_v2": pos_model_v2,
    "neg_model_v2": neg_model_v2,
    "pos_model": pos_model,
    "neg_model": neg_model,
    "NEGATION_WORDS": list(NEGATION_WORDS),
    "PUNCT_BOUNDARY": list(PUNCT_BOUNDARY),
    "POSITIVE_WORDS": list(POSITIVE_WORDS),
    "NEGATIVE_WORDS": list(NEGATIVE_WORDS),
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

import os
size_mb = os.path.getsize("model.pkl") / 1024 / 1024
print(f"✅ Exported model.pkl ({size_mb:.1f} MB)")
print("⬇️  Downloading to your computer...")

files.download("model.pkl")
