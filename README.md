# Domain-Adaptive Adversarial Robustness System for Intelligent Transportation AI pplications 
---

## 🚦 Why Robustness Matters in Transportation

In autonomous driving and traffic automation, AI misclassifications can lead to critical safety hazards:
- A subtly perturbed **Stop Sign** can be misclassified as a **Speed Limit 80** sign.
- Small digital or physical perturbations on roadside cameras can deceive vehicle perception pipelines.
- **AdverScan** benchmarks, scores, and hardens transportation models before deployment to ensure resilience against these threats.

---

## 🔑 Key Features

- **Domain-Specific Benchmarking**: Built-in support for transportation datasets such as **GTSRB** (German Traffic Sign Recognition Benchmark, 43 classes) and Vision Transformers (ViT) / CNN architectures.
- **White-Box Adversarial Attacks**: Implementations of industry-standard adversarial attack methods:
  - **FGSM** (*Fast Gradient Sign Method*)
  - **PGD** (*Projected Gradient Descent*)
  - **DeepFool** (*Minimal Perturbation Attack*)
  - **FAB** (*Fast Adaptive Boundary Attack*)
  - **C&W** (*Carlini & Wagner L₂ Attack*)
- **Vulnerability & Safety Metrics**:
  - Attack Success Rate (**ASR**)
  - Clean Accuracy vs. Adversarial Accuracy
  - Prediction Confidence Shift & Shannon Entropy
  - Perturbation Budgets ($L_0, L_2, L_\infty$ norms)
- **Model Hardening & Retesting**: Defenses such as adversarial training, input preprocessing, and randomized smoothing with automated post-defense re-evaluation.
- **Automated Security Reports**: Generates structured summaries detailing model weaknesses and mitigation recommendations.

---

## 🔄 Assessment Pipeline

```
Traffic Scene / Sign Input
          │
          ▼
   Model Ingestion
   (ViT / PyTorch Transportation Models)
          │
          ▼
   Adversarial Perturbation
   (FGSM, PGD, DeepFool, FAB, CW)
          │
          ▼
   Vulnerability & Robustness Scoring
   (ASR, Accuracy Degradation, Perturbation Norms)
          │
          ▼
   Defensive Hardening
   (Preprocessing, Adversarial Training)
          │
          ▼
   Re-testing & Verification
          │
          ▼
   Transportation Security Report
```

---

## 📁 Repository Structure

```text
AdverScan-Trasportation/
├── app/
│   ├── ingestion/              # Model loaders & PyTorch runtime adapters
│   ├── attack_engine/          # White-box adversarial attacks (FGSM, PGD, DeepFool, FAB, CW)
│   ├── evaluation/             # Transportation dataset loaders (GTSRB) & metrics
│   ├── vulnerability_analysis/ # Vulnerability scoring & degradation analysis
│   ├── hardening/              # Defensive hardening techniques
│   ├── retest/                 # Post-hardening verification pipeline
│   ├── report_generator/       # Automated audit report generation
│   ├── api/                    # REST API services
│   └── dashboard/              # Visualization dashboard
├── scripts/                    # Evaluation & validation runner scripts
└── tests/                      # Automated test suite
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended for GPU acceleration)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Khushi-234/AdverScan-Trasportation.git
cd AdverScan-Trasportation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Validate Transportation Model & Dataset

Verify the pipeline on the German Traffic Sign Recognition Benchmark (GTSRB) with a Vision Transformer (ViT):

```bash
python scripts/validate_gtsrb_model.py
```

### 4. Run Adversarial Attack Evaluation

Evaluate model vulnerability against adversarial perturbations (e.g., FGSM with $\epsilon = 8/255$):

```bash
python scripts/test_gtsrb_adversarial_attacks.py --samples 50 --attack fgsm --epsilon 0.03137
```

### 5. Run Test Suite

```bash
pytest tests/
```

---

## 📊 Core Metrics Summary

| Metric | Description | Transportation Impact |
| :--- | :--- | :--- |
| **Clean Accuracy** | Accuracy on unmodified traffic inputs | Baseline perception capability under normal driving conditions |
| **Adversarial Accuracy** | Accuracy under adversarial perturbation | Residual reliability when subjected to visual interference or attacks |
| **ASR (Attack Success Rate)** | % of inputs successfully manipulated | Probability of an autonomous system being deceived |
| **Perturbation Norm ($L_\infty$)** | Magnitude of maximum pixel deviation | Imperceptibility of the perturbation to human drivers |
| **Confidence Degradation** | Drop in target class softmax confidence | Risk of indecision or hesitation in autonomous vehicle control systems |
