import streamlit as st
import joblib
import re
import pdfplumber
import docx

vectorizer = joblib.load(r"D:\Desktop\cv ml\.ipynb_checkpoints\vectorizer.pkl")
le = joblib.load(r"D:\Desktop\cv ml\.ipynb_checkpoints\label_encoder.pkl")
model = joblib.load(r"D:\Desktop\cv ml\.ipynb_checkpoints\knn_model.pkl")
  
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*',' ',resumeText)
    resumeText = re.sub('RT|cc',' ',resumeText)
    resumeText = re.sub('#\S+',' ',resumeText)
    resumeText = re.sub('@\S+',' ',resumeText)
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

st.title("Resume Classification Demo")
st.write("You can either paste your resume text or upload a PDF/DOCX file.")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf","docx"])
resume_input = st.text_area("Or paste your resume text here:")

if st.button("Predict Category"):
    text_to_process = ""
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text_to_process += page.extract_text() + " "
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text_to_process += para.text + " "
    

    elif resume_input:
        text_to_process = resume_input
    
   
    else:
        st.warning("Please provide text or upload a file!")
        st.stop()

    cleaned = cleanResume(text_to_process)
    

    vectorized = vectorizer.transform([cleaned])
    

    prediction_num = model.predict(vectorized)[0]
    prediction_name = le.inverse_transform([prediction_num])[0]  
    st.success(f"The predicted category is: {prediction_name}")
    
 
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vectorized)[0]
        st.write("Prediction Probabilities:")
        prob_dict = {cat: f"{prob:.2f}" for cat, prob in zip(le.classes_, probs)}
        st.json(prob_dict)
