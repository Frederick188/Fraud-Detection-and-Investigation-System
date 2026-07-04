# 🛡️ AI-Powered Financial Fraud Detection & Investigation System

An end-to-end AI-powered financial fraud detection system that combines **Machine Learning**, **Explainable AI (LIME)**, **Retrieval-Augmented Generation (RAG)**, and **Large Language Models (Gemini)** to detect fraudulent transactions and generate human-readable investigation reports.

---

## 📌 Overview

Traditional fraud detection models often provide only a prediction (Fraud/Non-Fraud) without explaining the reasoning behind it. This project enhances transparency by integrating Explainable AI and Retrieval-Augmented Generation to produce professional fraud investigation reports grounded in both model predictions and financial regulations.

---

## 🚀 Features

* 📂 Upload financial transaction datasets (CSV)
* 🌐 Enrich transactions using:

  * IP Geolocation API
  * Currency Exchange API
* 🤖 Detect fraudulent transactions using a **Random Forest Classifier**
* 💡 Explain individual predictions using **LIME (Local Interpretable Model-Agnostic Explanations)**
* 📚 Retrieve relevant fraud guidelines using **RAG (Retrieval-Augmented Generation)**
* 🧠 Generate AI-powered fraud investigation reports using **Gemini API**
* 💬 Interactive Fraud Investigation Assistant built with **LangChain**
* 📊 User-friendly interface built with **Streamlit**

---

## 🏗️ System Architecture

```text
                    User Uploads CSV
                           │
                           ▼
                 Data Preprocessing
                           │
                           ▼
          API Enrichment (IP + Currency)
                           │
                           ▼
              Random Forest Classifier
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
    Fraud Prediction                 LIME Explanation
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                   Investigation Object
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
      ChromaDB Retriever            Gemini LLM
            │                             │
            └──────────────┬──────────────┘
                           ▼
      AI Fraud Investigation Report
```

---

## 🧠 Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Explainable AI

* LIME

### Generative AI

* Gemini API
* LangChain

### Retrieval-Augmented Generation

* ChromaDB
* Hugging Face Embeddings
* Sentence Transformers

### APIs

* IPGeolocation API
* CurrencyFreaks API

### Frontend

* Streamlit

---

## 📂 Project Structure

```text
Financial-Fraud-Detection/
│
├── app.py
├── build_db.py
├── requirements.txt
│
├── knowledge_base/
│   ├── visa_rules.pdf
│   ├── mastercard.pdf
│   ├── pci_dss.pdf
│   ├── rbi_guidelines.pdf
│   └── fraud_patterns.pdf
│
├── chroma_db/
│
└── src/
    ├── agent.py
    ├── explain.py
    ├── model.py
    ├── preprocess.py
    ├── prompts.py
    ├── rag.py
    ├── utils.py
    └── vectorstore.py
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/financial-fraud-detection.git
cd financial-fraud-detection
```

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

IPGEO_API_KEY=YOUR_IPGEOLOCATION_API_KEY

CURRENCY_API_KEY=YOUR_CURRENCYFREAKS_API_KEY
```

---

## ▶️ Build Vector Database

Before running the application, build the Chroma vector database.

```bash
python build_db.py
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔄 Workflow

1. Upload transaction dataset.
2. Select transaction range.
3. Enrich transactions using external APIs.
4. Train the Random Forest model.
5. Generate LIME explanations.
6. Ask the AI Fraud Investigation Assistant questions.
7. Receive grounded investigation reports using ML outputs and retrieved fraud knowledge.

---

## 💬 Example Questions

* Why was this transaction classified as fraud?
* Explain the LIME prediction.
* Which features contributed the most?
* Should this transaction be blocked?
* What PCI DSS guideline applies?
* What RBI recommendation is relevant?
* Generate a fraud investigation report.

---

## 📈 Key Components

### Machine Learning

* Random Forest Classifier for fraud prediction.

### Explainable AI

* LIME explains individual transaction predictions.

### Retrieval-Augmented Generation

* Retrieves fraud-related documents from ChromaDB.

### Large Language Model

* Gemini generates grounded investigation reports using:

  * Transaction details
  * Model prediction
  * Fraud probability
  * LIME explanation
  * Retrieved fraud knowledge

---

## 🎯 Future Improvements

* PDF investigation report generation
* User authentication
* Real-time transaction monitoring
* SHAP-based explainability
* Multi-model comparison
* Fraud trend dashboard
* Cloud deployment
* REST API support

---

## 📚 Learning Outcomes

* Machine Learning
* Explainable AI (LIME)
* Retrieval-Augmented Generation (RAG)
* LangChain
* Prompt Engineering
* Vector Databases
* Streamlit Application Development
* API Integration
* Financial Fraud Analytics

---

## 👨‍💻 Author

**Udit Ray**

B.Tech – Computer Science and Business Systems

---

## 📄 License

This project is intended for educational and portfolio purposes.
