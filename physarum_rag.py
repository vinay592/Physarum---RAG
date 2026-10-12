import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.linear_model import LogisticRegression

# =========================
# DEVICE CONFIGURATION
# =========================
DEVICE = 0 if torch.cuda.is_available() else -1

# =========================
# MODELS
# =========================
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli", device=DEVICE)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=DEVICE)

# =========================
# UTILS
# =========================
def normalize(x):
    x = np.array(x)
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-8)

def get_rank(scores, labels):
    sorted_idx = np.argsort(scores)[::-1]
    ranks = np.where(labels[sorted_idx] == 1)[0]
    return ranks[0] + 1 if len(ranks) > 0 else len(scores)

def compute_metrics(ranks):
    ranks = np.array(ranks)
    return {
        "R@1": np.mean(ranks <= 1),
        "R@3": np.mean(ranks <= 3),
        "R@5": np.mean(ranks <= 5),
        "MRR": np.mean(1 / ranks),
        "Avg Rank": np.mean(ranks)
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================
# CHUNKING
# =========================
def chunk_text(text, max_words=50, overlap=15):
    words = text.split()
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+max_words])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
    return chunks

# =========================
# SCORING
# =========================
def get_chunk_embeddings(chunks):
    return bi_encoder.encode(chunks, convert_to_numpy=True)

def get_similarity_scores(q, chunks):
    q_emb = bi_encoder.encode([q], convert_to_numpy=True)[0]
    c_emb = get_chunk_embeddings(chunks)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    c_emb = c_emb / (np.linalg.norm(c_emb, axis=1, keepdims=True) + 1e-8)
    return c_emb @ q_emb

def get_cross_scores(q, chunks):
    pairs = [[q, c] for c in chunks]
    return cross_encoder.predict(pairs)

# =========================
# NLI & QA EVIDENCE
# =========================
def get_nli_scores_batch(chunks, hypotheses, batch_size=16):
    inputs = [f"{c} </s> {h}" for c, h in zip(chunks, hypotheses)]
    outputs = nli_pipeline(
        inputs,
        batch_size=batch_size,
        truncation=True,
        top_k=None
    )
    scores = []
    for out in outputs:
        entail = 0.0
        if isinstance(out, list):
            for item in out:
                if isinstance(item, dict) and "ENTAIL" in item.get("label", "").upper():
                    entail = item.get("score", 0.0)
        elif isinstance(out, dict):
            if "ENTAIL" in out.get("label", "").upper():
                entail = out.get("score", 0.0)
        elif isinstance(out, str):
            if "ENTAIL" in out.upper():
                entail = 1.0
        scores.append(entail)
    return np.array(scores)

def get_evidence_scores_fast(query, chunks):
    answers = []
    confs = []
    for chunk in chunks:
        try:
            qa_out = qa_pipeline(question=query, context=chunk)
            answer = qa_out["answer"]
            conf = qa_out["score"]
            if conf < 0.3 or not answer.strip():
                answers.append("")
                confs.append(0.0)
            else:
                answers.append(answer)
                confs.append(conf)
        except Exception:
            answers.append("")
            confs.append(0.0)

    hypotheses = [
        f"The answer is {a}." if a != "" else ""
        for a in answers
    ]

    valid_idx = [i for i, a in enumerate(answers) if a != ""]
    nli_scores = np.zeros(len(chunks))

    if len(valid_idx) > 0:
        valid_chunks = [chunks[i] for i in valid_idx]
        valid_hyp = [hypotheses[i] for i in valid_idx]
        entail_scores = get_nli_scores_batch(valid_chunks, valid_hyp)
        for idx, score in zip(valid_idx, entail_scores):
            nli_scores[idx] = score * confs[idx]

    return nli_scores

# =========================
# PHYSARUM v5
# =========================
def physarum_v5(sim_matrix, ce_scores, nli_scores,
                k=10, alpha=10.0, tau=0.5,
                lam_pos=0.2, lam_neg=0.15):
    N = len(ce_scores)

    knn_mask = np.zeros_like(sim_matrix)
    for i in range(N):
        top_k_idx = np.argsort(sim_matrix[i])[-k:]
        knn_mask[i, top_k_idx] = 1
    knn_mask = np.maximum(knn_mask, knn_mask.T)
    sim_sparse = sim_matrix * knn_mask

    gate = 1 / (1 + np.exp(-alpha * (ce_scores - tau)))

    P_pos = np.zeros(N)
    P_neg = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if ce_scores[j] > ce_scores[i] + 0.05:
                P_pos[i] += sim_sparse[i, j] * ce_scores[j] * gate[i]
            if nli_scores[j] < 0.2:
                P_neg[i] += sim_sparse[i, j] * (0.5 - nli_scores[j]) * gate[i]

    return ce_scores + lam_pos * P_pos - lam_neg * P_neg

def get_physarum_scores_v5(q, chunks, sim, ce, nli):
    chunk_embs = get_chunk_embeddings(chunks)
    chunk_embs = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = chunk_embs @ chunk_embs.T
    sim_matrix = sim_matrix * sim[None, :]
    return physarum_v5(sim_matrix, ce, nli)

# =========================
# RAG DATASET
# =========================
def build_rag_dataset(data, num_distractors, distractor_pool):
    rag_data = []
    dropped = 0

    for sample in data:
        q = sample["question"]
        context = sample["context"]
        all_answers = sample["answers"]["text"]  # all annotator answers

        chunks = chunk_text(context)
        labels = [1 if any(ans in c for ans in all_answers) else 0 for c in chunks]

        if sum(labels) == 0:
            dropped += 1
            continue

        distractors = random.sample(distractor_pool, min(num_distractors, len(distractor_pool)))
        distractor_chunks = []
        for d in distractors:
            distractor_chunks.extend(chunk_text(d)[:1])

        chunks = chunks + distractor_chunks
        labels = labels + [0] * len(distractor_chunks)

        rag_data.append({
            "query": q,
            "chunks": chunks,
            "labels": labels
        })

    print(f"Dropped {dropped}/{len(data)} samples ({100*dropped/len(data):.1f}%)")
    return rag_data

# =========================
# EXPERIMENT
# =========================
def run_single_experiment(train_data, test_data, train_contexts, test_contexts, num_distractors, seed):
    random.seed(seed)
    np.random.seed(seed)

    train_rag = build_rag_dataset(train_data, num_distractors, distractor_pool=train_contexts)
    test_rag = build_rag_dataset(test_data, num_distractors, distractor_pool=test_contexts)

    # ---------------------
    # TRAIN FEATURES
    # ---------------------
    feature_sets = {
        "CE+NLI": [],
        "CE+NLI+Sim": [],
        "CE+Physarum": [],
        "NLI+Physarum": [],
        "CE+NLI+Physarum": [],
        "CE+NLI+Physarum+Sim": []
    }
    y_train = []

    for sample in tqdm(train_rag, desc=f"Train (d={num_distractors}, seed={seed})"):
        q = sample["query"]
        chunks = sample["chunks"]
        labels = np.array(sample["labels"])

        sim = normalize(get_similarity_scores(q, chunks))
        ce = normalize(get_cross_scores(q, chunks))

        mask = ce > 0.2
        nli_raw = np.zeros(len(chunks))
        if mask.sum() > 0:
            selected_chunks = [chunks[i] for i in range(len(chunks)) if mask[i]]
            selected_scores = get_evidence_scores_fast(q, selected_chunks)
            idx = 0
            for i in range(len(chunks)):
                if mask[i]:
                    nli_raw[i] = selected_scores[idx]
                    idx += 1

        nli = normalize(nli_raw)
        phy = get_physarum_scores_v5(q, chunks, sim=sim, ce=ce, nli=nli_raw)

        for i in range(len(chunks)):
            feature_sets["CE+NLI"].append([ce[i], nli[i]])
            feature_sets["CE+NLI+Sim"].append([ce[i], nli[i], sim[i]])
            feature_sets["CE+Physarum"].append([ce[i], phy[i]])
            feature_sets["NLI+Physarum"].append([nli[i], phy[i]])
            feature_sets["CE+NLI+Physarum"].append([ce[i], nli[i], phy[i]])
            feature_sets["CE+NLI+Physarum+Sim"].append([ce[i], nli[i], phy[i], sim[i]])
            y_train.append(labels[i])

    y_train = np.array(y_train)

    # ---------------------
    # TRAIN MODELS
    # ---------------------
    models = {}
    weights = {}
    for name, feats in feature_sets.items():
        X = np.array(feats)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X, y_train)
        models[name] = model
        weights[name] = model.coef_[0]

    # ---------------------
    # EVALUATION
    # ---------------------
    results = {
        "Sim": [], "CE": [], "NLI": [], "Physarum": [],
        "CE+NLI": [], "CE+NLI+Sim": [],
        "CE+Physarum": [], "NLI+Physarum": [],
        "CE+NLI+Physarum": [], "CE+NLI+Physarum+Sim": []
    }

    for sample in tqdm(test_rag, desc=f"Test (d={num_distractors}, seed={seed})"):
        q = sample["query"]
        chunks = sample["chunks"]
        labels = np.array(sample["labels"])

        sim = normalize(get_similarity_scores(q, chunks))
        ce = normalize(get_cross_scores(q, chunks))

        mask = ce > 0.2
        nli_raw = np.zeros(len(chunks))
        if mask.sum() > 0:
            selected_chunks = [chunks[i] for i in range(len(chunks)) if mask[i]]
            selected_scores = get_evidence_scores_fast(q, selected_chunks)
            idx = 0
            for i in range(len(chunks)):
                if mask[i]:
                    nli_raw[i] = selected_scores[idx]
                    idx += 1

        nli = normalize(nli_raw)
        phy = get_physarum_scores_v5(q, chunks, sim=sim, ce=ce, nli=nli_raw)

        results["Sim"].append(get_rank(sim, labels))
        results["CE"].append(get_rank(ce, labels))
        results["NLI"].append(get_rank(nli, labels))
        results["Physarum"].append(get_rank(phy, labels))

        feat_map = {
            "CE+NLI": np.stack([ce, nli], axis=1),
            "CE+NLI+Sim": np.stack([ce, nli, sim], axis=1),
            "CE+Physarum": np.stack([ce, phy], axis=1),
            "NLI+Physarum": np.stack([nli, phy], axis=1),
            "CE+NLI+Physarum": np.stack([ce, nli, phy], axis=1),
            "CE+NLI+Physarum+Sim": np.stack([ce, nli, phy, sim], axis=1),
        }

        for name, feat in feat_map.items():
            scores = models[name].predict_proba(feat)[:, 1]
            results[name].append(get_rank(scores, labels))

    metrics = {k: compute_metrics(v) for k, v in results.items()}
    return metrics, weights

# =========================
# MAIN ENTRYPOINT
# =========================
def main():
    SEEDS = [42, 199, 69]
    DISTRACTORS = [30, 50]

    # Load dataset
    print("Loading SQuAD v2 dataset...")
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")
    dataset = dataset.filter(lambda x: len(x['answers']['text']) > 0)
    dataset = dataset.select(range(1000))

    train_data = dataset.select(range(800))
    test_data = dataset.select(range(800, 1000))

    train_contexts = [d["context"] for d in train_data]
    test_contexts = [d["context"] for d in test_data]

    all_results = {}

    for num_distractors in DISTRACTORS:
        print(f"\n{'='*20} DISTRACTORS = {num_distractors} {'='*20}")
        seed_metrics = []

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            metrics, weights = run_single_experiment(
                train_data, test_data,
                train_contexts, test_contexts,
                num_distractors, seed
            )
            seed_metrics.append(metrics)

            # Per-seed table
            print(f"\n| Method | R@1 | R@3 | R@5 | MRR | Avg Rank |")
            print(f"|--------|-----|-----|-----|-----|----------|")
            for method, m in metrics.items():
                print(f"| {method} | {m['R@1']:.3f} | {m['R@3']:.3f} | {m['R@5']:.3f} | {m['MRR']:.3f} | {m['Avg Rank']:.2f} |")

            print("\nWeights:")
            for name, w in weights.items():
                print(f"  {name} {np.round(w, 3)}")

        # Average across seeds
        print(f"\n=== AVERAGE RESULTS (d={num_distractors}) ===")
        print(f"\n| Method | R@1 | R@3 | R@5 | MRR | Avg Rank |")
        print(f"|--------|-----|-----|-----|-----|----------|")
        methods = list(seed_metrics[0].keys())
        for method in methods:
            avg = {k: np.mean([s[method][k] for s in seed_metrics]) for k in ["R@1", "R@3", "R@5", "MRR", "Avg Rank"]}
            print(f"| {method} | {avg['R@1']:.3f} | {avg['R@3']:.3f} | {avg['R@5']:.3f} | {avg['MRR']:.3f} | {avg['Avg Rank']:.2f} |")

        all_results[num_distractors] = seed_metrics


if __name__ == "__main__":
    main()
