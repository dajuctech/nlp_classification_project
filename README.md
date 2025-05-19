# 🧠 NLP Classification Project

A production-ready Natural Language Processing (NLP) pipeline for text classification tasks, built with Scikit-learn and deployed across major cloud platforms including Azure ML, AWS SageMaker, and Google Vertex AI. This project supports full model lifecycle from training and evaluation to API and UI deployment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📁 Project Structure

```
NLP_CLASSIFICATION_PROJECT/
├── api/                    # Flask API for model inference
│   └── app.py
├── cloud/                  # Cloud deployment notebooks
│   ├── aws_sagemaker.ipynb
│   ├── azure_ml_deploy.ipynb
│   └── gcp_vertex_ai.ipynb
├── dashboard/              # Streamlit dashboard
│   └── streamlit_app.py
├── data/                   # Raw or processed data (optional)
├── models/                 # Saved model and preprocessing artifacts
│   ├── final_model.pkl
│   ├── label_encoder.pkl
│   └── vectorizer.pkl
├── notebooks/              # Jupyter notebooks for exploration
├── outputs/                # Evaluation metrics or model outputs
├── src/                    # Source code for model training
├── tests/                  # Unit tests
├── run_pipeline.py         # Pipeline to train and save the model
├── config.yaml             # Configuration file for model training
├── Dockerfile              # Dockerfile for local containerization
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🚀 Features

- End-to-end pipeline from preprocessing to deployment
- Modular codebase with configurable YAML
- Streamlit-based UI for human interaction
- REST API using FastAPI/Flask
- Multi-cloud deployment support
- Pretrained model artifacts stored in \`data/models/\`

---

## 🛠️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/nlp-classification-project.git
cd nlp-classification-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Running the App

### API (FastAPI or Flask)

```bash
python api/app.py
```

### Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## ☁️ Cloud Deployment

Deployment notebooks are available under the \`cloud/\` directory:

- **Azure ML**: `azure_ml_deploy.ipynb`
- **AWS SageMaker**: `aws_sagemaker.ipynb`
- **GCP Vertex AI**: `gcp_vertex_ai.ipynb`

Each notebook contains code to register and deploy the model on the respective platform.

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📌 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 💡 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature requests or improvements.
