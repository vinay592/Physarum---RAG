# Physarum-RAG: Biologically-Inspired Graph Reranking for Retrieval-Augmented Generation

A biologically-inspired passage reranking framework for Retrieval-Augmented Generation (RAG) that models chunk networks using principles derived from *Physarum polycephalum* (slime mold) flow dynamics, coupled with Cross-Encoder scoring and Natural Language Inference (NLI) evidence verification.

---

## 📌 Overview

In dense retrieval and RAG pipelines, isolated passage scoring often struggles when multiple distracting passages share lexical or topical overlap with a user's query.

*Physarum polycephalum* is a single-celled organism capable of solving optimal transport network problems (such as maze routing and recreating transport infrastructure) by reinforcing nutrient-rich pathways while pruning unproductive tubes.

**Physarum-RAG** adapts this mechanism to document chunk reranking:
1. **Network Construction**: Builds a $k$-nearest neighbors ($k$-NN) semantic similarity graph among retrieved chunks, weighted by query relevance.
2. **Confidence Gating**: Employs a sigmoid gate parameterized by Cross-Encoder scores to control flow resistance.
3. **Flow Propagation ($P_{pos}$)**: Propagates positive reinforcement from high-confidence neighboring chunks to relevant adjacent context.
4. **Distractor Suppression ($P_{neg}$)**: Identifies weakly entailed candidate chunks via Question Answering (QA) and Natural Language Inference (NLI), penalizing misleading paths.
5. **Ensemble Stacking**: Trains balanced Logistic Regression models on combined multi-modal retrieval features (Similarity, Cross-Encoder, NLI, and Physarum scores).

---

## 🏗️ Architecture & Model Pipeline

```text
Query + Context Chunks + Distractor Chunks
       │
       ├──► Bi-Encoder (all-MiniLM-L6-v2) ──► Semantic Embeddings & k-NN Graph
       │
       ├──► Cross-Encoder (ms-marco-MiniLM-L-6-v2) ──► Relevance Scores & Gating
       │
       └──► Evidence Verification:
             ├── QA Model (deepset/roberta-base-squad2) ──► Extracted Answers & Confidence
             └── NLI Model (facebook/bart-large-mnli)    ──► Entailment Verification
                         │
                         ▼
             Physarum v5 Graph Flow Engine
            [Positive Flow + Negative Penalty]
                         │
                         ▼
           Logistic Regression Feature Stacking
                         │
                         ▼
             Final Ranked Passages (R@K, MRR)
```

### Models Used
- **Bi-Encoder**: `sentence-transformers/all-MiniLM-L6-v2`
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **QA Reader**: `deepset/roberta-base-squad2`
- **NLI Verifier**: `facebook/bart-large-mnli`

---

## 🧮 Physarum v5 Mathematical Formulation

For $N$ candidate chunks:
1. **Sparsified $k$-NN Graph**:
   $$\text{Mask}_{i, j} = \max(\mathbb{I}[j \in \text{top}_k(\text{Sim}_i)], \mathbb{I}[i \in \text{top}_k(\text{Sim}_j)])$$
   $$W_{i, j} = (\mathbf{c}_i \cdot \mathbf{c}_j) \times \text{Sim}(q, c_j) \times \text{Mask}_{i, j}$$

2. **Gating Function**:
   $$\text{Gate}_i = \sigma(\alpha \cdot (\text{CE}_i - \tau))$$
   *(Default: $\alpha = 10.0$, $\tau = 0.5$)*

3. **Positive and Negative Potentials**:
   $$P_{pos}(i) = \sum_{j \neq i, \text{CE}_j > \text{CE}_i + 0.05} W_{i, j} \cdot \text{CE}_j \cdot \text{Gate}_i$$
   $$P_{neg}(i) = \sum_{j \neq i, \text{NLI}_j < 0.2} W_{i, j} \cdot (0.5 - \text{NLI}_j) \cdot \text{Gate}_i$$

4. **Updated Score**:
   $$\text{Score}_{\text{Physarum}}(i) = \text{CE}_i + \lambda_{pos} P_{pos}(i) - \lambda_{neg} P_{neg}(i)$$
   *(Default: $\lambda_{pos} = 0.2$, $\lambda_{neg} = 0.15$)*

---

## 📂 Project Structure

```text
physarum/
├── physarum_rag.py    # Main implementation of chunking, scoring, Physarum v5, and evaluation
├── main.py            # Entrypoint script
├── requirements.txt   # Python package dependencies
└── README.md          # Documentation and methodology
```

---

## 🚀 Installation & Setup

### 1. Clone or navigate to the repository
```bash
cd /path/to/physarum
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note**: For GPU acceleration (CUDA), install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/). The code automatically detects and routes transformer pipelines to GPU (`device = 0`) when available.

---

## 💻 Running the Experiment

Run the full benchmark suite:

```bash
python main.py
```
*or*
```bash
python physarum_rag.py
```

### Experiment Configuration
- **Dataset**: `rajpurkar/squad_v2` (validation set, answerable questions)
- **Train / Test Split**: 800 train samples / 200 test samples
- **Distractor Counts**: Evaluated across `[30, 50]` distractors per query
- **Evaluation Seeds**: `[42, 199, 69]`
- **Evaluation Metrics**:
  - `R@1`, `R@3`, `R@5` (Recall at rank $k$)
  - `MRR` (Mean Reciprocal Rank)
  - `Avg Rank`

---

## 📊 Feature Configurations Evaluated

The benchmark compares standalone retrievers against stacked Logistic Regression combinations:

| Method | Description |
|---|---|
| **Sim** | Normalized Bi-Encoder cosine similarity |
| **CE** | Normalized Cross-Encoder score |
| **NLI** | Evidence-verified NLI entailment score |
| **Physarum** | Biologically-inspired Physarum v5 rerank score |
| **CE + NLI** | Logistic Regression combining Cross-Encoder and NLI |
| **CE + NLI + Sim** | Logistic Regression combining CE, NLI, and Bi-Encoder Similarity |
| **CE + Physarum** | Ensemble combining Cross-Encoder and Physarum v5 |
| **NLI + Physarum** | Ensemble combining NLI evidence and Physarum v5 |
| **CE + NLI + Physarum** | Multi-signal ensemble fusing CE, NLI, and Physarum |
| **CE + NLI + Physarum + Sim** | Full feature ensemble combining all signals |

---

## 📄 License

MIT License.
# Physarum---RAG
