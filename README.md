# Resume Classification System

A machine learning-based system that automatically classifies resumes into different job categories using Natural Language Processing (NLP) and K-Nearest Neighbors (KNN) algorithm.

# Features

- **Multi-format Support**: Upload PDF or DOCX files, or paste resume text directly
- **Accurate Classification**: Uses TF-IDF vectorization and KNN classifier
- **Interactive UI**: Built with Streamlit for easy use
- **Real-time Predictions**: Get instant category predictions with probability scores

# Technologies Used

- **Python 3.12**
- **Scikit-learn**: Machine Learning models
- **Streamlit**: Web interface
- **Pandas**: Data manipulation
- **pdfplumber**: PDF text extraction
- **python-docx**: DOCX file handling
- **Joblib**: Model serialization

# Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Vectorization**: TF-IDF with 1500 features
- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%

# Installation

# 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/resume-classifier.git
cd resume-classifier
```

# 2. Install dependencies
```bash
pip install -r requirements.txt
```

# 3. Run the application
```bash
streamlit run resume.py
```

# Required Packages

Create a `requirements.txt` file with:
```
streamlit==1.31.0
scikit-learn==1.4.0
pandas==2.2.0
pdfplumber==0.10.3
python-docx==1.1.0
joblib==1.3.2
```

# Project Structure
```
resume-classifier/
│
├── resume.py                 # Streamlit application
├── train_model.ipynb         # Model training notebook
├── .ipynb_checkpoints/       # Saved models
│   ├── vectorizer.pkl
│   ├── knn_model.pkl
│   └── label_encoder.pkl
├── data/
│   └── UpdatedResumeDataSet.csv
├── requirements.txt
└── README.md
```

# How to Use

1. **Run the application**:
```bash
   streamlit run resume.py
```

2. **Upload a resume** (PDF/DOCX) or **paste text**

3. **Click "Predict Category"**

4. **View results** with probability scores for each category

# Screenshots

![App Interface](screenshots/app_screenshot.png)
*Main interface of the Resume Classifier*

# Supported Categories

- Data Science
- Web Development
- Mobile Development
- Database Administration
- Network Security
- Business Analyst
- *[Add your actual categories]*

# Model Training

To retrain the model:

1. Open `train_model.ipynb`
2. Run all cells
3. The trained models will be saved in `.ipynb_checkpoints/`

#  Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 99% |
| Test Accuracy | 98% |
| F1-Score | 0.97 |

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Basmala Mohammed**
- GitHub: [@Basmala Mohammed](https://github.com/basmalamhammed)
- LinkedIn: [Basmala Mohamed](https://www.linkedin.com/in/basmala-mohammed-8103572b4?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

# Acknowledgments

- Dataset source: [Kaggle Resume Dataset](https://www.kaggle.com/)
- Inspired by NLP and ML community



 If you find this project helpful, please give it a star!
```

---

## requirements.txt


requirements.txt`:
```
streamlit==1.31.0
scikit-learn==1.4.0
pandas==2.2.0
pdfplumber==0.10.3
python-docx==1.1.0
joblib==1.3.2
numpy==1.26.3
```

---

## .gitignore

`.gitignore` :
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Jupyter Notebook
.ipynb_checkpoints

# Data files 
*.csv
data/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
