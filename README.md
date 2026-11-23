# AdverScan 🛡️

**AdverScan** is framework for **Adversarial Hardening, Vulnerability Analysis, and Security Testing** of machine learning models across generalized domains: **Computer Vision (Images)**, **Natural Language Processing (Text)**, **Tabular Data**, and **Time-Series Sequences**.

---

## 🌟 Key Capabilities

- **Multi-Modal Model Ingestion**: Native adapters for **PyTorch**, **Hugging Face Transformers**, and **Scikit-learn** models across vision, NLP, tabular, and sequence classification.
- **Comprehensive Adversarial Attack Engine**: Implements white-box attack vectors including FGSM, PGD, C&W, DeepFool, HotFlip, HopSkipJump, Temporal PGD, and Capital PGD.
- **Quantitative Vulnerability Scoring**: Automated computation of Attack Success Rate (ASR), accuracy degradation, stealth metrics, and weighted composite vulnerability scores (0–100).
- **Root-Cause Explainability (XAI)**: Integrated interpretability via SHAP, LIME, Integrated Gradients, and Saliency maps to diagnose model weaknesses under perturbation.
- **Context-Aware Hardening Engine**: Dynamic selection and tuning of defense techniques—including Confidence-Based Rejection, Adversarial Detection, Randomized Smoothing, Spatial Filtering, Feature Squeezing, and Model Ensembles.
- **Empirical Retesting & Verification**: Strict pre- vs. post-defense re-testing ensuring empirical robustness improvements without degrading clean accuracy or violating latency budgets.
- **Automated Audit Reporting**: Automatic compilation and export of comprehensive security audit reports in **PDF (via ReportLab)**, **JSON**, and **Markdown** formats.
- **Persistent Assessment History & Comparison**: Built-in SQLite database tracking historical scans, multi-attack evaluations, defense trials, and side-by-side assessment comparisons.
- **Interactive Streamlit Dashboard**: Full-featured web interface featuring real-time evaluation progress, interactive attack parameter tuning, XAI visualizations, and one-click report downloads.

---

## 🏗️ Project Architecture

```text
AdverScan/
│
├── app/                              # Core Security & Evaluation Framework
│   ├── ingestion/                    # Model & dataset loading (PyTorch, HuggingFace, Scikit-learn)
│   ├── attack_engine/                # Adversarial attack implementations across domains
│   ├── evaluation/                   # Robustness metrics, confidence extraction, & loaders
│   ├── vulnerability_analysis/       # Scoring engine, degradation metrics, & threat matrices
│   ├── explainability/               # Root-cause attribution & interpretability (SHAP, LIME, IG)
│   ├── hardening/                    # Defense algorithms (rejection, detection, ensembles, smoothing)
│   ├── retest/                       # Empirical hardening verification & threshold compliance
│   ├── report_generator/             # Automated multi-format reporting (PDF via ReportLab, JSON, MD)
│   ├── storage/                      # SQLite ORM storage repository, scan history & artifact tracking
│   ├── orchestration/                # Pipeline execution engine & step coordinators
│   ├── api/                          # FastAPI REST endpoints
│   ├── core/                         # Core domain schemas, exceptions, & base interfaces
│   └── utils/                        # System utilities, logging, & helpers
│
├── frontend/                         # Modern Interactive Streamlit Application
│   ├── streamlit_app.py              # Main dashboard application entrypoint
│   ├── components/                   # UI components (history view, comparison, metrics, sidebar, charts, XAI)
│   ├── services/                     # Frontend orchestration services (ingestion, attack, defense, report)
│   ├── styles/                       # CSS themes (glassmorphic dark theme, animations, layout)
│   └── utils/                        # Frontend state helpers & formatting utilities
│
├── storage_data/                     # Persistent local database & artifact storage
│   ├── adverscan.db                  # Central SQLite database tracking all scans & executions
│   ├── artifacts/                    # Saved evaluation artifacts (clean inputs, adversarial tensors)
│   └── backups/                      # Database backups
│
├── configs/                          # System configurations & presets
├── data/                             # Benchmark test datasets (tabular, CSV, etc.)
├── models/                           # Pre-trained model weights & checkpoints (.pt)
├── reports/                          # Audit output reports (JSON, Markdown, PDF)
├── results/                          # Raw evaluation logs & telemetry metrics
├── scripts/                          # CLI automation scripts, benchmarks, & evaluation runners
└── tests/                            # Unit and integration test suites
```

---

## 🔄 End-to-End Evaluation Pipeline

The framework executes an integrated 8-stage security assessment lifecycle:

1. **Stage 1 — Model & Dataset Ingestion**: Load and standardize target models, processors, tokenizers, and reference datasets across supported domains.
2. **Stage 2 — Baseline Robustness Evaluation**: Profile clean classification accuracy, top-1/top-2 prediction margins, and baseline inference latency.
3. **Stage 3 — Adversarial Threat Simulation**: Generate adversarial perturbations using targeted or untargeted attack vectors calibrated to the data domain.
4. **Stage 4 — Vulnerability Assessment**: Measure post-attack accuracy, Attack Success Rate (ASR), perturbation magnitudes ($L_\infty$, $L_2$), and calculate composite risk ratings.
5. **Stage 5 — Root-Cause Explainability (XAI)**: Compute feature attributions and saliency shifts between clean and perturbed samples to pinpoint model vulnerabilities.
6. **Stage 6 — Defensive Hardening**: Apply context-recommended defenses (e.g. rejection filters, input transforms, adversarial detectors) to harden candidate models.
7. **Stage 7 — Retesting & Compliance**: Re-evaluate hardened candidates against adversarial test vectors to verify empirical robustness recovery and clean accuracy preservation.
8. **Stage 8 — Security Audit Reporting**: Compile findings into a structured audit report with risk ratings, vulnerability breakdown, defense efficacy, and downloadable PDF.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.14)
- Virtual environment tool (`venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/adver-scan-framework.git
   cd adver-scan-framework
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Or install development dependencies in editable mode:*
   ```bash
   make dev-install
   ```

---

## 🖥️ Running AdverScan

### 1. Launch the Streamlit Web Application
Run the interactive web dashboard to configure scans, run attacks, inspect XAI attributions, execute defenses, and view scan history:
```bash
streamlit run frontend/streamlit_app.py
```
*Or using the Makefile shortcut:*
```bash
make run-ui
```
Open your browser at `http://localhost:8501`.

### 2. Launch the FastAPI Backend
Start the REST API server:
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```
*Or using the Makefile shortcut:*
```bash
make run-api
```
API documentation is available at `http://localhost:8000/docs`.

---

## 🧪 Running Tests

Execute the comprehensive test suite across core modules, hardening defenses, re-testing, reporting, and frontend services:
```bash
pytest tests/
```
*Or with test coverage:*
```bash
make test
```

---

## ⚙️ Makefile Reference

| Command            | Description                                              |
|--------------------|----------------------------------------------------------|
| `make install`     | Install production dependencies from `requirements.txt`  |
| `make dev-install` | Install package in editable mode with development tools  |
| `make run-ui`      | Launch the Streamlit web dashboard                       |
| `make run-api`     | Start the FastAPI backend server                         |
| `make test`        | Run all unit and integration tests with coverage         |
| `make format`      | Format code using `black` and `isort`                    |
| `make lint`        | Run static code analysis (`flake8`, `mypy`)              |
| `make clean`       | Remove cache directories (`__pycache__`, `.pytest_cache`)|
