# WORKS!!!

import numpy as np
import pandas as pd
from hmmlearn.hmm import MultinomialHMM

import time


# =======================
# Data
# =======================

BRCA1_START = 43043692
BRCA1_END   = 43170845

CLIN_BENIGN = "clinvar_result_benign.txt"
CLIN_PATH   = "clinvar_result_pathogenic.txt"

RANDOM_STATE = 42

BIO_MUT_TYPES = {
    "synonymous": 0,
    "missense": 1,
    "nonsense": 2,
    "frameshift": 3,
    "inframe_indel": 4,
    "canonical_splice": 5,
    "splice_region": 6,
    "utr": 7,
    "intronic_deep": 8,
    "microsatellite": 9,
    "structural_cnv": 10,
}

# Swapping to using given information in the "Molecular Consequence" column - have issues with missing data
SO_TO_BIO = {
    "synonymous_variant": "synonymous",
    "missense_variant": "missense",
    "stop_gained": "nonsense",
    "frameshift_variant": "frameshift",
    "inframe_insertion": "inframe_indel",
    "inframe_deletion": "inframe_indel",

    "splice_acceptor_variant": "canonical_splice",
    "splice_donor_variant": "canonical_splice",
    "splice_region_variant": "splice_region",

    "5_prime_UTR_variant": "utr",
    "3_prime_UTR_variant": "utr",

    "intron_variant": "intronic_deep",

    # fallback categories
    "protein_altering_variant": "missense",
    "coding_sequence_variant": "missense",

    "structural_variant": "structural_cnv",
}


N_CATEGORIES = len(BIO_MUT_TYPES)

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
        cats = [classify_mutation_biotype(s["spdi"]) for s in samples]
    
    print(sorted(set(cats)))
    return samples


# =======================
# CLASSIFICATION LOGIC 
# =======================

def classify_mutation_biotype(spdi):
    """Assign biologically meaningful mutation type category."""
    deleted = spdi["deleted"]
    inserted = spdi["inserted"]
    del_len = len(deleted)
    ins_len = len(inserted)

    # Structural CNVs ≥50 bp
    if del_len >= 50 or ins_len >= 50:
        return BIO_MUT_TYPES["structural_cnv"]

    # Microsatellite: repeated strings like "CA", etc
    bases = {"A", "C", "G", "T"}
    if all(c in bases for c in deleted + inserted):
        if (len(deleted) >= 6 and len(set(deleted)) == 1) or \
           (len(inserted) >= 6 and len(set(inserted)) == 1):
            return BIO_MUT_TYPES["microsatellite"]

    # Possible synonymous/missense/nonsense
    if del_len == 1 and ins_len == 1:
        ref = deleted.upper()
        alt = inserted.upper()
        if ref in bases and alt in bases:
            # This is an approximation.

            # Stop codon creation ("nonsense") if alt is ʻ*ʻ
            if alt == "*":
                return BIO_MUT_TYPES["nonsense"]

            # canʻt compute amino acids without coding context.
            # so...
            # fallback...
            if ref == alt:
                return BIO_MUT_TYPES["synonymous"]

            return BIO_MUT_TYPES["missense"]

    # Frameshift or in frame
    if (ins_len - del_len) % 3 == 0:
        return BIO_MUT_TYPES["inframe_indel"]
    else:
        return BIO_MUT_TYPES["frameshift"]


# =======================
# HMM TRAINING
# =======================

def build_sequences(samples):
    X = []
    L = []

    for s in samples:
        cat = classify_mutation_biotype(s["spdi"])

        onehot = np.zeros(N_CATEGORIES, dtype=int)
        onehot[cat] = 1

        X.append(onehot)
        L.append(1)

    return np.array(X, dtype=int), L


def safe_fix_hmm(hmm):
    """Ensure valid probability distributions."""
    n = hmm.n_components

    # startprob_
    if not hasattr(hmm, "startprob_"):
        hmm.startprob_ = np.ones(n) / n
    s = hmm.startprob_.sum()
    if s == 0:
        hmm.startprob_ = np.ones(n) / n
    else:
        hmm.startprob_ /= s

    # transmat_
    if not hasattr(hmm, "transmat_"):
        hmm.transmat_ = np.ones((n, n)) / n
    rowsum = hmm.transmat_.sum(axis=1, keepdims=True)
    hmm.transmat_ = np.divide(
        hmm.transmat_, rowsum,
        out=np.ones_like(hmm.transmat_) / n,
        where=rowsum != 0
    )
    return hmm


def train_biotype_hmms():
    samples = load_clinvar(CLIN_BENIGN, CLIN_PATH)
    samples = [s for s in samples if s["label"] in (0, 1)]

    labels = [s["label"] for s in samples]

    from sklearn.model_selection import train_test_split
    train_s, test_s = train_test_split(
        samples, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
    )

    train_b = [s for s in train_s if s["label"] == 0]
    train_p = [s for s in train_s if s["label"] == 1]

    Xb, Lb = build_sequences(train_b)
    Xp, Lp = build_sequences(train_p)

    hmm_b = MultinomialHMM(
        n_components=1,
        n_iter=50,
        random_state=RANDOM_STATE,
        init_params='e',   # only train emissions
        params='e'
    )
    hmm_b.n_features = N_CATEGORIES 

    hmm_p = MultinomialHMM(
        n_components=1,
        n_iter=50,
        random_state=RANDOM_STATE,
        init_params='e',
        params='e'
    )
    hmm_p.n_features = N_CATEGORIES 

    if len(Xb) > 0:
        hmm_b.fit(Xb, lengths=Lb)
    else:
        hmm_b.startprob_ = np.array([1.0])
        hmm_b.transmat_ = np.array([[1.0]])
        hmm_b.emissionprob_ = np.ones((1, N_CATEGORIES)) / N_CATEGORIES

    if len(Xp) > 0:
        hmm_p.fit(Xp, lengths=Lp)
    else:
        hmm_p.startprob_ = np.array([1.0])
        hmm_p.transmat_ = np.array([[1.0]])
        hmm_p.emissionprob_ = np.ones((1, N_CATEGORIES)) / N_CATEGORIES

    hmm_b = safe_fix_hmm(hmm_b)
    hmm_p = safe_fix_hmm(hmm_p)

    print(np.unique(Xb))
    print(np.unique(Xp))

    print("Benign Transition Matrix:")
    print(hmm_b.transmat_)
    print("\nBenign Emission Matrix:")
    print(hmm_b.emissionprob_)
    print("\nPath Transition Matrix:")
    print(hmm_p.transmat_)
    print("\nPath Emission Matrix:")
    print(hmm_p.emissionprob_)
    

    return hmm_b, hmm_p, test_s


# =======================
# EVALUATE
# =======================

def evaluate_biotype_hmm():
    hmm_b, hmm_p, test = train_biotype_hmms()

    y_true = []
    y_pred = []

    for s in test:
        y_true.append(s["label"])
        cat = classify_mutation_biotype(s["spdi"])
        onehot = np.zeros((1, N_CATEGORIES), dtype=int)
        onehot[0, cat] = 1

        lb = hmm_b.score(onehot)
        lp = hmm_p.score(onehot)

        y_pred.append(1 if lp > lb else 0)

    from sklearn.metrics import classification_report, accuracy_score
    print("\n=== Type-Only HMM Classification ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    start_time = time.time()
    evaluate_biotype_hmm()
    end_time = time.time()
    print(f"Total Runtime: {end_time - start_time:.2f} seconds")
