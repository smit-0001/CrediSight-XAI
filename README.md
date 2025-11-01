# 🧠 CrediSight XAI Engine

> **📈 A Production-Grade, Containerized MLOps Pipeline for Credit Risk Prediction with Explainable AI**

This is a complete, end-to-end MLOps portfolio project that demonstrates a **production-grade system** for a critical business problem — predicting **credit default risk** and, more importantly, providing **human-readable explanations** for each decision to satisfy regulatory requirements (like **XAI**).

---

## 🎬 Demo

<!-- DEMO GIF  -->

<p align="center">
  <img src="[PLACEHOLDER: Add demo GIF here. Use 'ScreenToGif' to record testing the 3 FastAPI endpoints with sample data.]" alt="CrediSight Demo GIF">
</p>

---

## 🏷️ Badges

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/FastAPI-0.100.0-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-build-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/XAI-SHAP-purple.svg" alt="SHAP for XAI">
  <img src="https://img.shields.io/badge/ML-XGBoost-orange.svg" alt="XGBoost">
</p>

---

## 🎯 The “Why” — Beyond Just a Model

> **The Black Box Problem:** A simple “denied” from a model is not acceptable. This engine provides a **“glass box”** by using **SHAP** to explain why a decision was made — critical for **regulatory compliance** and **customer trust**.

**Deployment & Portability:** Everything, including models and dependencies, is containerized with **Docker**. The result is a single, portable, and reproducible “walking skeleton.”

**Maintainability:** Built with **FastAPI** using a clean *router vs. service* separation for easy upgrades.

**Model Comparison:** Exposes endpoints for both **Logistic Regression** and **XGBoost**, demonstrating readiness for **A/B testing**.

---

## 🛠️ Tech Stack & Architecture

| Category                 | Tools                      |
| ------------------------ | -------------------------- |
| **Core Framework**       | Python 3.10                |
| **ML & Preprocessing**   | scikit-learn, XGBoost      |
| **Explainable AI (XAI)** | SHAP (TreeExplainer)       |
| **API & Server**         | FastAPI, Uvicorn           |
| **Containerization**     | Docker                     |
| **Data Handling**        | Pandas, NumPy              |
| **Development**          | VS Code, GitHub Codespaces |



## 🔮 Work in Progress / Future Improvements

* [ ] **LLM:** Adding a Generative AI whcih feeds on SHAP output to provide a more human friendly summary .
* [ ] **CI/CD Pipeline:** Automating `rebuild.sh` with GitHub Actions.
* [ ] **Experiment Tracking:** Integrate MLflow for metrics & versioning.

---

## 🚀 Getting Started

You can run this project locally using **Docker**.

#### 🧩 Prerequisites

> You must have **Docker** and **Git** installed on your system.

#### 1️⃣ Clone the Repository

```bash
git clone [YOUR_REPOSITORY_URL]
cd CrediSight-XAI
```

#### 2️⃣ Run the Training Pipeline

```bash
pip install -r requirements.txt
python train.py
```

This creates artifacts:

* `logistic_model.joblib`
* `preprocessor_pipeline.joblib`
* `xgb_model.json`

All saved in `/artifacts`.

### 3️⃣ Build & Run the Docker Container

```bash
chmod +x rebuild.sh
./rebuild.sh
```

This script will:

* Stop & remove old containers
* Build a new image `credisight-api`
* Run it on **localhost:8080**

#### 4️⃣ Use the API

Open your browser to **[http://localhost:8080/docs](http://localhost:8080/docs)** for interactive Swagger docs.

---

## 🔌 Endpoints (API Reference)

### Prediction Endpoints

#### `POST /predict/logistic`

> Returns a **probability of default** from Logistic Regression.

#### `POST /predict/xgb`

> Returns a **probability of default** from XGBoost.

### XAI Endpoint (The “Why”)

#### `POST /explain/xgb`

> Provides a **human-readable explanation** of a prediction using SHAP.

Example Request:

```json
{
  "ExternalRiskEstimate": 55,
  "MSinceOldestTradeOpen": 144,
  "MSinceMostRecentTradeOpen": 4,
  "AverageMInFile": 84,
  "NumSatisfactoryTrades": 20,
  "NumTrades60Ever2DerogPubRec": 3,
  "NumTrades90Ever2DerogPubRec": 0,
  "PercentTradesNeverDelq": 83,
  "MSinceMostRecentDelq": 2,
  "MaxDelq2PublicRecLast12M": 3,
  "MaxDelqEver": 5,
  "NumTotalTrades": 23,
  "NumTradesOpeninLast12M": 1,
  "PercentInstallTrades": 43,
  "MSinceMostRecentInqexcl7days": 0,
  "NumInqLast6M": 0,
  "NumInqLast6Mexcl7days": 0,
  "NetFractionRevolvingBurden": 33,
  "NetFractionInstallBurden": null,
  "NumRevolvingTradesWBalance": 8,
  "NumInstallTradesWBalance": 1,
  "NumBank2NatlTradesWHighUtilization": 1,
  "PercentTradesWBalance": 69
}
```

Example Response:

```json
{
  "base_value": 0.0128603745,
  "explanations": [
    {"feature": "ExternalRiskEstimate", "shap_value": 1.3609},
    {"feature": "PercentTradesNeverDelq", "shap_value": 0.5015},
    {"feature": "MaxDelq2PublicRecLast12M", "shap_value": 0.3655}
  ],
  "summary": "This is a high-risk profile, primarily driven by: ExternalRiskEstimate, PercentTradesNeverDelq, and MaxDelq2PublicRecLast12M."
}
```
---

## 👤 Author

**[Smit Dhandhukia]**
🔗 [LinkedIn](https://www.linkedin.com/in/smit-b-dhandhukia/)
💻 [GitHub](https://github.com/smit-0001)

---