# Adversarial Robustness & Security Assessment Platform 🛡️

An automated framework for **Adversarial Testing, Vulnerability Scoring, and Defensive Hardening** of Computer Vision AI/ML models for transportaion domain.

---

## 🌟 Key Capabilities

- **Computer Vision Model Ingestion**: Native adapters for **PyTorch**, **Torchvision**, **Hugging Face Vision Transformers**, and **ONNX** image classifiers.
- **Adversarial Attack Engine**: Implements white-box adversarial attacks specifically calibrated for image data:
  - **FGSM** (Fast Gradient Sign Method)
  - **PGD** (Projected Gradient Descent - $L_\infty$ and $L_2$)
  - **DeepFool** (Minimal Perturbation Boundary Attack)
  - **C&W** (Carlini & Wagner $L_2$ Optimization Attack)
  - **FAB** (Fast Adaptive Boundary Attack)
- **Quantitative Vulnerability Scoring**: Automated computation of:
  - Attack Success Rate (ASR)
  - Clean vs. Adversarial Accuracy Drop
  - Perturbation Magnitude Budgets ($L_\infty$, $L_2$, Mean Perturbation)
  - Composite Robustness & Vulnerability Score (0–100)
  - Compliance mapping to **MITRE ATLAS**, **NIST AI RMF**, and **OWASP ML Top 10**
- **Image-Centric Defensive Hardening**: Context-aware defense algorithms for computer vision pipelines:
  - **Adversarial Training**: Multi-step adversarial batch generation and robust fine-tuning
  - **Spatial Smoothing**: Gaussian blur and median spatial filtering to neutralize high-frequency perturbations
  - **Feature Squeezing**: Bit-depth reduction and colour depth quantization
  - **Randomized Smoothing**: Certified defense via isotropic Gaussian noise injection
  - **Data Augmentation**: Robust image transformations to reinforce decision boundaries
- **Empirical Retesting & Verification**: Pre-defense vs. post-defense evaluation measuring empirical robustness gain, clean accuracy preservation, and inference latency impacts.
- **Automated Audit Reporting**: Generation and export of security audit reports in **PDF (via ReportLab)**, **JSON**, and **Markdown** formats.
- **Persistent Assessment History & Comparison**: Embedded SQLite storage for auditing historical scans, retests, defense evaluations, and side-by-side comparative analysis.
- **Interactive Streamlit Dashboard**: Full-featured web interface featuring real-time attack execution, perturbation visual inspection, defense configuration, and audit report generation.

---

## 🏗️ Project Architecture

```text
├── app/                              # Core Security & Evaluation Framework
│   ├── ingestion/                    # Vision model & dataset loading (PyTorch, Torchvision, Hugging Face)
│   ├── attack_engine/                # Adversarial attack implementations (FGSM, PGD, DeepFool, C&W, FAB)
│   ├── evaluation/                   # Robustness metrics, classification metrics, & dataset factory
│   ├── vulnerability_analysis/       # Scoring engine, degradation metrics, & threat matrices
│   ├── hardening/                    # Image defense techniques (Adversarial Training, Spatial Smoothing, etc.)
│   ├── retest/                       # Empirical hardening verification & threshold compliance
│   ├── report_generator/             # Automated multi-format reporting (PDF, JSON, Markdown)
│   ├── storage/                      # SQLite ORM storage repository, scan history & artifact tracking
│   ├── orchestration/                # Pipeline execution engine & workflow coordinators
│   ├── api/                          # FastAPI REST endpoints
│   ├── core/                         # Core domain schemas, exceptions, & base interfaces
│   └── utils/                        # System utilities, device management (CUDA/CPU), & logging
│
├── frontend/                         # Interactive Streamlit Web Application
│   ├── streamlit_app.py              # Main dashboard application entrypoint
│   ├── components/                   # UI components (sidebar, charts, metrics, history, comparison)
│   ├── services/                     # Frontend service layer (ingestion, attack, defense, reports, storage)
│   ├── styles/                       # CSS themes (glassmorphic dark theme, animations, layout)
│   └── utils/                        # Frontend state helpers & validation utilities
│
├── storage_data/                     # Persistent local database & artifact storage
│   ├── robustness_assessment.db      # SQLite database tracking all scans & executions
│   └── artifacts/                    # Saved evaluation artifacts (clean inputs, adversarial tensors)
│
├── configs/                          # System configurations & presets
├── models/                           # Model weights & vision checkpoints
├── reports/                          # Audit output reports (JSON, Markdown, PDF)
├── results/                          # Raw evaluation logs & telemetry metrics
├── scripts/                          # Automation scripts, benchmarks, & evaluation runners
└── tests/                            # Unit and integration test suites
```

---

## 🔄 End-to-End Evaluation Pipeline

The platform follows a 7-stage security assessment and hardening lifecycle:

1. **Stage 1 — Model & Dataset Ingestion**: Load and standardize the target vision model, preprocessing transforms (normalization, resize), and validation image dataset.
2. **Stage 2 — Baseline Robustness Evaluation**: Profile clean classification accuracy, top-1/top-2 prediction confidence, and baseline inference latency.
3. **Stage 3 — Adversarial Threat Simulation**: Generate adversarial perturbations using selected white-box attack vectors (FGSM, PGD, DeepFool, C&W, FAB) across configurable perturbation budgets ($\epsilon$).
4. **Stage 4 — Vulnerability Assessment & Scoring**: Measure post-attack accuracy, Attack Success Rate (ASR), perturbation distortion, and compute weighted vulnerability scores (0–100) mapped to MITRE ATLAS and NIST AI RMF.
5. **Stage 5 — Defensive Hardening**: Apply image-specific hardening defenses (Adversarial Training, Spatial Smoothing, Feature Squeezing, Randomized Smoothing, or Data Augmentation).
6. **Stage 6 — Retesting & Robustness Verification**: Re-evaluate the hardened model against the adversarial attack suite to measure empirical robustness recovery, clean accuracy retention, and latency tradeoffs.
7. **Stage 7 — Security Certification & Audit Reporting**: Compile findings into an executive-ready audit report with vulnerability rankings, defense performance, and compliance ratings (downloadable as PDF, Markdown, or JSON).

In addition, the platform includes:
- **Scan History**: Browse, inspect, or delete previous security scans stored in SQLite.
- **Comparative Analysis**: Compare any two scans (e.g. baseline vs. defended model) with side-by-side metric deltas and radar visualizations.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (tested on Python 3.10, 3.11, 3.12, 3.13)
- PyTorch 2.x with CUDA (optional, CPU execution supported)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Platform

### 1. Launch the Streamlit Web Application
Run the interactive security dashboard:
```bash
streamlit run frontend/streamlit_app.py
```
Or with explicit Python executable:
```bash
python -m streamlit run frontend/streamlit_app.py
```
Access the application in your browser at `http://localhost:8501`.

### 2. Launch the FastAPI Backend (Optional)
If running headless or via REST endpoints:
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation is available at `http://localhost:8000/docs`.

---

## 🧪 Running Tests

Execute the unit and integration test suites:
```bash
pytest tests/
```
Run with coverage reporting:
```bash
pytest --cov=app tests/
```
