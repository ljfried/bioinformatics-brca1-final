# WORKS!!!

import numpy as np
import pandas as pd
from hmmlearn.hmm import MultinomialHMM

# =======================
# Data
# =======================

BRCA1_START = 43043692
BRCA1_END   = 43170845

BRCA1_FASTA = "Homo_sapiens_BRCA1_sequence.fa"

CLIN_BENIGN = "clinvar_result_benign.txt"
CLIN_PATH   = "clinvar_result_pathogenic.txt"

RANDOM_STATE = 42

# How many discrete bins along the gene
N_LOC_BINS = 100

# =======================
# UTILS
# =======================

def parse_spdi(spdi):
    if not isinstance(spdi, str):
        return None
    parts = spdi.split(":")
    if len(parts) != 4:
        return None
    seqid, pos, deleted, inserted = parts
    try:
        pos_i = int(pos)
    except:
        return None
    return {"chrom": seqid, "pos": pos_i, "deleted": deleted, "inserted": inserted}

def get_mutation_pos_from_spdi(spdi):
    if spdi is None:
        return None
    pos = spdi["pos"]
    if not (BRCA1_START <= pos < BRCA1_END):
        return None
    return pos - BRCA1_START

def normalize_label(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"benign", "likely benign"}:
        return 0
    if s in {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}:
        return 1
    return None

def load_clinvar(ben_file, path_file):
    cols = ["Canonical SPDI", "Germline classification"]
    b = pd.read_csv(ben_file, sep="\t", low_memory=False)[cols]
    p = pd.read_csv(path_file, sep="\t", low_memory=False)[cols]
    b["label"] = 0
    p["label"] = 1
    df = pd.concat([b, p], ignore_index=True)

    samples = []
    for _, row in df.iterrows():
        spdi_raw = row["Canonical SPDI"]
        if not isinstance(spdi_raw, str):
            continue
        parsed = parse_spdi(spdi_raw)
        if parsed is None:
            continue
        label = normalize_label(row.get("label"))
        samples.append({"spdi": parsed, "label": label})
    
    return samples

# =======================
# LOCATION SEQUENCES
# =======================

def get_location_bin(pos):
    """Map a genomic position to a discrete bin index."""
    rel_pos = get_mutation_pos_from_spdi(pos)
    if rel_pos is None:
        return None
    bin_idx = int(rel_pos / (BRCA1_END - BRCA1_START) * N_LOC_BINS)
    return min(bin_idx, N_LOC_BINS - 1)

def build_location_sequences(samples):
    X = []
    L = []

    for s in samples:
        bin_idx = get_location_bin(s["spdi"])
        if bin_idx is None:
            continue
        onehot = np.zeros(N_LOC_BINS, dtype=int)
        onehot[bin_idx] = 1
        X.append(onehot)
        L.append(1)

    return np.array(X, dtype=int), L

def safe_fix_hmm(hmm):
    """Ensure valid probability distributions."""
    n = hmm.n_components

    if not hasattr(hmm, "startprob_"):
        hmm.startprob_ = np.ones(n) / n
    s = hmm.startprob_.sum()
    hmm.startprob_ /= s if s != 0 else 1

    if not hasattr(hmm, "transmat_"):
        hmm.transmat_ = np.ones((n, n)) / n
    rowsum = hmm.transmat_.sum(axis=1, keepdims=True)
    hmm.transmat_ = np.divide(
        hmm.transmat_, rowsum,
        out=np.ones_like(hmm.transmat_) / n,
        where=rowsum != 0
    )
    return hmm

# =======================
# TRAINING
# =======================

def train_location_hmms():
    samples = load_clinvar(CLIN_BENIGN, CLIN_PATH)
    samples = [s for s in samples if s["label"] in (0, 1)]
    labels = [s["label"] for s in samples]

    from sklearn.model_selection import train_test_split
    train_s, test_s = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
    )

    train_b = [s for s in train_s if s["label"] == 0]
    train_p = [s for s in train_s if s["label"] == 1]

    Xb, Lb = build_location_sequences(train_b)
    Xp, Lp = build_location_sequences(train_p)

    hmm_b = MultinomialHMM(n_components=1, n_iter=50, random_state=RANDOM_STATE, init_params='e', params='e')
    hmm_b.n_features = N_LOC_BINS

    hmm_p = MultinomialHMM(n_components=1, n_iter=50, random_state=RANDOM_STATE, init_params='e', params='e')
    hmm_p.n_features = N_LOC_BINS

    if len(Xb) > 0:
        hmm_b.fit(Xb, lengths=Lb)
    else:
        hmm_b.startprob_ = np.array([1.0])
        hmm_b.transmat_ = np.array([[1.0]])
        hmm_b.emissionprob_ = np.ones((1, N_LOC_BINS)) / N_LOC_BINS

    if len(Xp) > 0:
        hmm_p.fit(Xp, lengths=Lp)
    else:
        hmm_p.startprob_ = np.array([1.0])
        hmm_p.transmat_ = np.array([[1.0]])
        hmm_p.emissionprob_ = np.ones((1, N_LOC_BINS)) / N_LOC_BINS

    hmm_b = safe_fix_hmm(hmm_b)
    hmm_p = safe_fix_hmm(hmm_p)

    return hmm_b, hmm_p, test_s

# =======================
# EVALUATION
# =======================

def evaluate_location_hmm():
    hmm_b, hmm_p, test = train_location_hmms()

    y_true = []
    y_pred = []

    for s in test:
        bin_idx = get_location_bin(s["spdi"])
        if bin_idx is None:
            continue
        y_true.append(s["label"])
        onehot = np.zeros((1, N_LOC_BINS), dtype=int)
        onehot[0, bin_idx] = 1

        lb = hmm_b.score(onehot)
        lp = hmm_p.score(onehot)

        y_pred.append(1 if lp > lb else 0)

    from sklearn.metrics import classification_report, accuracy_score
    print("\n=== Location-Based HMM Classification ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    evaluate_location_hmm()
