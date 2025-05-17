# 🩺 AI Medical Diagnosis Assistant

A Streamlit web application that uses a combination of Azure OpenAI and a local machine learning model to help users understand and explore possible medical conditions based on their self‑reported symptoms. This tool is for prognosis & informational purposes only and does **not** provide a professional medical diagnosis.

---

## 🚀 Key Features

* **Symptom Extraction**: Leverages Azure OpenAI to parse free‑text user input and extract known symptom keywords.
* **Interactive Clarification**: Dynamically ask follow‑up questions to confirm or rule out additional symptoms.
* **Local ML Model**: Uses a pre‑trained model (loaded from Azure Blob Storage) to compute likelihoods for various conditions.
* **Azure OpenAI Interpretation**: Generates a friendly, easy‑to‑understand summary and advice based on model outputs.
* **Azure Integration**:

  * **OpenAI Service** for LLM integration.
  * **Blob Storage** for artifact (model + JSON) hosting.
  * **Azure App Service** deployment with custom startup script.

---

## 📦 Architecture Overview

```text
User Browser  ↔  Streamlit App (App Service)
      ↕            │
      ↕            ├─ Azure OpenAI (chat completion)
      ↕            ├─ Local ML Model (joblib)
      ↕            └─ Azure Blob Storage (model/artifacts)
```

---

## 📋 Prerequisites

1. **Azure Subscription** with:

   * An **App Service** (Linux, Python 3.10+, B1+ tier).
   * A **Storage Account** with a private Blob container for your artifacts.
   * An **Azure OpenAI** resource & deployment.

2. **Local / Dev Machine**:

   * Python 3.10+ installed
   * Azure CLI (`az`) and/or VS Code with Azure extensions for deployment

---

## 🛠️ Setup & Configuration

1. **Clone the repo**:

   ```bash
   git clone https://github.com/pranaydeep139/AI-Medical-Diagnosis.git
   cd azure_ai_med
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\\Scripts\\activate    # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Streamlit secrets**:
   Populate your `~/.streamlit/secrets.toml` with:

   ```toml
   AZURE_OPENAI_KEY = "<your-openai-key>"
   AZURE_OPENAI_ENDPOINT = "<your-openai-endpoint>"
   AZURE_OPENAI_DEPLOYMENT_NAME = "<your-ai-model-name>"

   AZURE_STORAGE_CONNECTION_STRING = ""
   AZURE_STORAGE_ACCOUNT_NAME = ""
   ```

5. **Run locally**:

   ```bash
   python train_model.py # (for training the model and loading the required files - only once)
   streamlit run app.py
   ```

---

## ☁️ Deploy to Azure App Service

1. **Ensure** you have a `run.sh` startup file:

   ```bash
   #!/bin/bash
   python -m streamlit run your_app.py \
     --server.port 8000 \
     --server.address 0.0.0.0
   ```

2. **Set executable**:

   ```bash
   chmod +x run.sh
   ```

3. **Configure in Azure Portal**:

   * Go to your App Service → **Configuration** → **General settings**.
   * **Startup Command**: `run.sh`

4. **Deploy**:

   * Via **VS Code**: Right‑click your App Service → **Deploy to Web App** → select folder.
   * Or **Azure CLI**:

     ```bash
     az webapp up \
       --name <app-name> \
       --resource-group <rg> \
       --runtime "PYTHON:3.10" \
       --sku B1 \
       --src-path .
     az webapp config set --name <app-name> --resource-group <rg> --startup-file run.sh
     ```

5. **Browse** to `https://<app-name>.azurewebsites.net`.

---

## 📈 Monitoring & Logs

* **Log Stream**: In Azure Portal → App Service → **Log stream** to see `print()` output live.
---

## ⚠️ Disclaimer

This application is for **prognosis** and **informational purposes only** and does not constitute professional medical advice. Always consult a qualified healthcare professional for full diagnosis and treatment.
