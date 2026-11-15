"""DeepDTA (Davis) backend: FastAPI + ONNX Runtime (no TensorFlow needed)."""
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent

# ---- Load settings and model once, when the server starts ----
cfg = json.loads((BASE / "config.json").read_text())
SMILES_VOCAB = cfg["smiles_vocab"]
PROTEIN_VOCAB = cfg["protein_vocab"]
MAX_SMILES = cfg["max_smiles_length"]      # 85
MAX_PROTEIN = cfg["max_protein_length"]    # 1200

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1              # keeps memory use low on small servers
session = ort.InferenceSession(
    str(BASE / "deepdta_davis.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
)

# Weak / moderate / strong bands for Davis pKd.
# 7 is the paper's "binds" cutoff (Section 3.4). The 6 edge is our own choice.
WEAK_BELOW = 6.0
STRONG_FROM = 7.0

app = FastAPI(title="DeepDTA binding affinity")


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=2000)
    protein: str = Field(..., min_length=1, max_length=6000)


def clean_smiles(text: str) -> str:
    return re.sub(r"\s+", "", text)


def clean_protein(text: str) -> str:
    # drop FASTA header lines (starting with ">"), spaces and line breaks
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith(">")]
    return re.sub(r"\s+", "", "".join(lines)).upper()


def encode(sequence: str, vocab: dict, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.int64)    # 0 = padding
    for i, ch in enumerate(sequence[:max_len]):
        out[i] = vocab.get(ch, 0)
    return out


def label_for(pkd: float) -> str:
    if pkd < WEAK_BELOW:
        return "Weak"
    if pkd < STRONG_FROM:
        return "Moderate"
    return "Strong"


@app.post("/predict")
def predict(req: PredictRequest):
    smiles = clean_smiles(req.smiles)
    protein = clean_protein(req.protein)
    warnings = []

    if not smiles or not protein:
        return {"error": "Please enter both a drug SMILES string and a protein sequence."}

    bad_s = sorted({c for c in smiles if c not in SMILES_VOCAB})
    bad_p = sorted({c for c in protein if c not in PROTEIN_VOCAB})
    if bad_s:
        msg = (
            "The model has not seen these SMILES characters and ignored them: "
            + " ".join(bad_s) + ". The prediction may be unreliable."
        )
        if any(c in "cnosp[]@+-/\\" for c in bad_s):
            msg += (
                " The model was trained on SMILES written with capital letters only and no "
                "square brackets (for example C1=CC=CC=C1 instead of c1ccccc1). "
                "Please rewrite the drug in that style."
            )
        warnings.append(msg)
    if bad_p:
        warnings.append(
            "The model has not seen these protein letters and ignored them: "
            + " ".join(bad_p) + ". The prediction may be unreliable."
        )
    if len(smiles) > MAX_SMILES:
        warnings.append(f"The SMILES is longer than {MAX_SMILES} characters, so only the first {MAX_SMILES} were used.")
    if len(protein) > MAX_PROTEIN:
        warnings.append(f"The protein is longer than {MAX_PROTEIN} letters, so only the first {MAX_PROTEIN} were used.")

    d = encode(smiles, SMILES_VOCAB, MAX_SMILES)[None, :]
    p = encode(protein, PROTEIN_VOCAB, MAX_PROTEIN)[None, :]
    pkd = float(session.run(None, {"drug_input": d, "protein_input": p})[0][0][0])

    return {
        "dataset": "Davis",
        "value_name": "pKd",
        "value": round(pkd, 3),
        "label": label_for(pkd),
        "warnings": warnings,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")
