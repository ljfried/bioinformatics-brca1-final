import numpy as np
import pandas as pd
import time
import argparse

from hmmlearn.hmm import MultinomialHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# =======================
# Data
# =======================

DEFAULT_BENIGN = "clinvar_result_benign.txt"
DEFAULT_PATH   = "clinvar_result_pathogenic.txt"

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

N_TYPE_CATEGORIES = len(BIO_MUT_TYPES)

N_LOC_BINS = 300

# =======================
# HELPER
# =======================

def safe_fix_hmm(hmm):
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
# DATA PREPARATION
# =======================

class DataPreparation:
    def __init__(self, benign_file=DEFAULT_BENIGN, path_file=DEFAULT_PATH):
        self.benign_file = benign_file
        self.path_file = path_file
        self.loc_start = None
        self.loc_end = None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def classify_mutation_biotype(spdi):
        deleted = spdi["deleted"]
        inserted = spdi["inserted"]
        del_len = len(deleted)
        ins_len = len(inserted)

        bases = {"A", "C", "G", "T"}

        # Structural CNVs ≥50 bp
        if del_len >= 50 or ins_len >= 50:
            return BIO_MUT_TYPES["structural_cnv"]

        # Microsatellite: single base repeated many times
        if all(c in bases for c in deleted + inserted):
            if (len(deleted) >= 6 and len(set(deleted)) == 1) or \
               (len(inserted) >= 6 and len(set(inserted)) == 1):
                return BIO_MUT_TYPES["microsatellite"]

        # SNV-like
        if del_len == 1 and ins_len == 1:
            ref = deleted.upper()
            alt = inserted.upper()

            if alt == "*":
                return BIO_MUT_TYPES["nonsense"]

            if ref == alt:
                return BIO_MUT_TYPES["synonymous"]

            return BIO_MUT_TYPES["missense"]

        # Frameshift vs in-frame
        if (ins_len - del_len) % 3 == 0:
            return BIO_MUT_TYPES["inframe_indel"]
        else:
            return BIO_MUT_TYPES["frameshift"]

    def load(self):
        """Load and parse ClinVar files and return samples."""
        cols = ["Canonical SPDI", "Germline classification"]
        b = pd.read_csv(self.benign_file, sep="\t", low_memory=False)[cols]
        p = pd.read_csv(self.path_file, sep="\t", low_memory=False)[cols]
        b["label"] = 0
        p["label"] = 1

        df = pd.concat([b, p], ignore_index=True)

        samples = []
        positions = []
        for _, row in df.iterrows():
            spdi_raw = row["Canonical SPDI"]
            parsed = self.parse_spdi(spdi_raw)
            if parsed is None:
                continue

            label = self.normalize_label(row.get("label"))
            samples.append({"spdi": parsed, "label": label})
            positions.append(parsed["pos"])
        
        if positions:
            self.loc_start = min(positions)
            self.loc_end = max(positions) + 1  # half-open interval

        return samples


# =======================
# TYPE-BASED HMM
# =======================

class TypeHMM:
    def __init__(self, data_prep: DataPreparation):
        self.data_prep = data_prep

    def build_sequences(self, samples):
        X, L = [], []
        for s in samples:
            cat = self.data_prep.classify_mutation_biotype(s["spdi"])
            onehot = np.zeros(N_TYPE_CATEGORIES, dtype=int)
            onehot[cat] = 1
            X.append(onehot)
            L.append(1)
        return np.array(X, dtype=int), L

    def train(self):
        samples = self.data_prep.load()
        samples = [s for s in samples if s["label"] in (0, 1)]
        labels = [s["label"] for s in samples]

        train_s, test_s = train_test_split(
            samples, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
        )

        train_b = [s for s in train_s if s["label"] == 0]
        train_p = [s for s in train_s if s["label"] == 1]

        Xb, Lb = self.build_sequences(train_b)
        Xp, Lp = self.build_sequences(train_p)

        hmm_b = MultinomialHMM(
            n_components=1,
            n_iter=50,
            init_params='e', params='e',
            random_state=RANDOM_STATE
        )
        hmm_b.n_features = N_TYPE_CATEGORIES

        hmm_p = MultinomialHMM(
            n_components=1,
            n_iter=50,
            init_params='e', params='e',
            random_state=RANDOM_STATE
        )
        hmm_p.n_features = N_TYPE_CATEGORIES

        if len(Xb) > 0:
            hmm_b.fit(Xb, lengths=Lb)
        else:
            hmm_b.startprob_ = np.array([1.0])
            hmm_b.transmat_ = np.array([[1.0]])
            hmm_b.emissionprob_ = np.ones((1, N_TYPE_CATEGORIES)) / N_TYPE_CATEGORIES

        if len(Xp) > 0:
            hmm_p.fit(Xp, lengths=Lp)
        else:
            hmm_p.startprob_ = np.array([1.0])
            hmm_p.transmat_ = np.array([[1.0]])
            hmm_p.emissionprob_ = np.ones((1, N_TYPE_CATEGORIES)) / N_TYPE_CATEGORIES

        hmm_b = safe_fix_hmm(hmm_b)
        hmm_p = safe_fix_hmm(hmm_p)

        return hmm_b, hmm_p, test_s

    def evaluate(self):
        hmm_b, hmm_p, test = self.train()

        y_true = []
        y_pred = []

        for s in test:
            y_true.append(s["label"])
            cat = self.data_prep.classify_mutation_biotype(s["spdi"])
            onehot = np.zeros((1, N_TYPE_CATEGORIES), dtype=int)
            onehot[0, cat] = 1

            lb = hmm_b.score(onehot)
            lp = hmm_p.score(onehot)

            y_pred.append(1 if lp > lb else 0)

        print("\n=== Type-Only HMM Classification ===")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


# =======================
# LOCATION-BASED HMM
# =======================

class LocationHMM:
    def __init__(self, data_prep: DataPreparation):
        self.data_prep = data_prep

    def bin_location(self, spdi):
        pos = spdi["pos"]

        start = self.data_prep.loc_start
        end = self.data_prep.loc_end

        if start is None or end is None:
            return None

        if not (start <= pos < end):
            return None

        rel = pos - start
        span = end - start

        bin_idx = int(rel / span * N_LOC_BINS)
        return min(bin_idx, N_LOC_BINS - 1)

    def build_sequences(self, samples):
        X, L = [], []
        for s in samples:
            bin_idx = self.bin_location(s["spdi"])
            if bin_idx is None:
                continue
            onehot = np.zeros(N_LOC_BINS, dtype=int)
            onehot[bin_idx] = 1
            X.append(onehot)
            L.append(1)
        return np.array(X, dtype=int), L

    def train(self):
        samples = self.data_prep.load()
        samples = [s for s in samples if s["label"] in (0, 1)]
        labels = [s["label"] for s in samples]

        train_s, test_s = train_test_split(
            samples, test_size=0.3, stratify=labels, random_state=RANDOM_STATE
        )

        train_b = [s for s in train_s if s["label"] == 0]
        train_p = [s for s in train_s if s["label"] == 1]

        Xb, Lb = self.build_sequences(train_b)
        Xp, Lp = self.build_sequences(train_p)

        hmm_b = MultinomialHMM(
            n_components=1, n_iter=50, random_state=RANDOM_STATE,
            init_params='e', params='e'
        )
        hmm_b.n_features = N_LOC_BINS

        hmm_p = MultinomialHMM(
            n_components=1, n_iter=50, random_state=RANDOM_STATE,
            init_params='e', params='e'
        )
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

    def evaluate(self):
        hmm_b, hmm_p, test = self.train()

        y_true = []
        y_pred = []

        for s in test:
            bin_idx = self.bin_location(s["spdi"])
            if bin_idx is None:
                continue
            y_true.append(s["label"])

            onehot = np.zeros((1, N_LOC_BINS), dtype=int)
            onehot[0, bin_idx] = 1

            lb = hmm_b.score(onehot)
            lp = hmm_p.score(onehot)
            y_pred.append(1 if lp > lb else 0)

        print("\n=== Location-Only HMM Classification ===")
        print(classification_report(y_true, y_pred, zero_division=0))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


# =======================
# MAIN
# =======================

def main():
    full_time = time.time()
    parser = argparse.ArgumentParser(description="HMM Models")
    parser.add_argument("--mode", choices=["type", "location", "both"], default="both", help='Which model(s) to run: "type", "location", or "both"')
    parser.add_argument("--benign", default=DEFAULT_BENIGN, help="Benign file from ClinVar")
    parser.add_argument("--path", default=DEFAULT_PATH, help="Pathogenic file from ClinVar")
    args = parser.parse_args()

    data = DataPreparation(args.benign, args.path)

    if args.mode in ("type", "both"):
        print("\nRunning TYPE model...")
        t0 = time.time()
        TypeHMM(data).evaluate()
        print(f"Type runtime: {time.time() - t0:.2f} seconds")

    if args.mode in ("location", "both"):
        print("\nRunning LOCATION model...")
        t0 = time.time()
        LocationHMM(data).evaluate()
        print(f"Location runtime: {time.time() - t0:.2f} seconds")
    
    print(f"Full runtime: {time.time() - full_time:.2f} seconds")


if __name__ == "__main__":
    main()