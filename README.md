# Transportation AI Adversarial Robustness System 🛡️

A domain-focused framework for **adversarial testing and robustness assessment of Artificial Intelligence (AI) models used in intelligent transportation applications**.

The project focuses on evaluating the robustness of **image classification and computer vision models** against adversarial attacks using transportation-related datasets and scenarios.

## 🏗️ Project Architecture

```text
app/
│
├── model_ingestion/          # Model loading, validation, and preparation
│
├── attack_engine/            # Adversarial attack implementations
│   └── image/                # Image-based white-box attacks
│       ├── fgsm.py
│       ├── pgd.py
│       ├── deepfool.py
│       └── fab.py
│
├── vulnerability_assessment/ # Adversarial vulnerability analysis
│
├── scoring/                  # Robustness and vulnerability scoring
│
├── hardening/                # Model defense and hardening techniques
│
├── retest/                   # Re-evaluation after hardening
│
├── report_generator/         # Vulnerability and experiment reports
│
├── configs/                  # Dataset, model, attack, and experiment configurations
│
├── reports/                  # Generated assessment reports
│
├── results/                  # Raw attack and evaluation results
│
├── tests/                    # Unit and integration tests
│
└── scripts/                  # Utility and experiment scripts
```

## 🎯 Project Scope

The project follows an end-to-end adversarial robustness assessment workflow:

```text
Model & Dataset
      ↓
Model Ingestion
      ↓
Adversarial Attack
      ↓
Vulnerability Assessment
      ↓
Robustness Scoring
      ↓
Model Hardening
      ↓
Re-testing
      ↓
Security Report
```

The current implementation primarily targets **image-based AI models used in intelligent transportation applications**.

## 🔬 Adversarial Attacks

The image attack module focuses on white-box adversarial attacks, including:

* **FGSM** — Fast Gradient Sign Method
* **PGD** — Projected Gradient Descent
* **DeepFool** — Minimal perturbation-based attack
* **FAB** — Fast Adaptive Boundary attack

These attacks are used to evaluate how model predictions change when carefully crafted adversarial perturbations are introduced into transportation images.

## 📊 Vulnerability Assessment

The assessment module evaluates model behavior under clean and adversarial conditions using metrics such as:

* Attack Success Rate (ASR)
* Clean accuracy
* Adversarial accuracy
* Confidence change
* Perturbation magnitude
* L₀, L₂, and L∞ norms
* Model performance degradation

The results are used to characterize the model's adversarial vulnerability.

## 🛡️ Model Hardening

The hardening module evaluates defense mechanisms intended to improve model robustness against adversarial inputs.

Depending on the selected experiment, defenses may include:

* Input preprocessing
* Randomized smoothing
* Adversarial training
* Other applicable robustness techniques

The hardened model is subsequently passed to the re-testing stage to determine whether its vulnerability has changed.

## 🔄 Re-testing

After applying a defense, the model is evaluated again using adversarial attacks.

```text
Original Model
      ↓
Attack
      ↓
Vulnerability Assessment
      ↓
Hardening
      ↓
Re-test
      ↓
Compare Results
```

This allows the effectiveness of the applied hardening technique to be assessed using the same evaluation metrics.

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* PyTorch
* Required project dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/Khushi-234/AdverScan-Transportation.git
cd AdverScan-Transportation
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Usage

The project can be used to:

1. Load and validate transportation image datasets.
2. Ingest compatible image classification models.
3. Execute selected white-box adversarial attacks.
4. Measure adversarial vulnerability.
5. Calculate robustness and degradation metrics.
6. Apply selected hardening techniques.
7. Re-test the hardened model.
8. Generate an assessment report.


## 📚 Research Focus

The project focuses on **adversarial robustness of AI models in intelligent transportation systems**, with particular emphasis on:

* Traffic sign recognition
* Transportation image classification
* Computer vision models
* Adversarial attack evaluation
* Model vulnerability assessment
* AI model hardening
* Post-defense robustness evaluation
