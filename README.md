# Disease-Prediction
Real-time disease diagnosis system using Big Data &amp; IoT. Predicts multiple diseases via ML models (Random Forest, FJAPSO optimization). Includes Streamlit UI for live predictions, patient risk reports, and interpretable results. Designed for scalable, privacy-aware healthcare diagnostics.
# 🩺 Big Data Analytics in IoT for Disease Diagnosis

This project delivers a machine learning-powered diagnostic system that uses **IoT sensor data** and **Big Data Analytics** to predict diseases in real time. Featuring a custom optimization algorithm (**FJAPSO**) and a user-friendly **Streamlit** interface, it enables efficient, accurate, and interpretable healthcare diagnostics.

---

## 🔍 Features
- Predicts Skin Cancer, HIV, Wilson Disease, Bone Disease, and Depression
- Uses real-time biometric and behavioral health data
- Confidence-based predictions
- Optimized with FJAPSO (Fusion Joint Adaptive PSO)
- Secure and scalable architecture
- Interactive UI with Streamlit

---

## 🧰 Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, joblib, Streamlit
- **Models**: Random Forest, Decision Tree
- **Optimization**: FJAPSO
- **Interface**: Streamlit App

---

## 🗂 Project Structure
```
├── data/                   # Sample patient reports
├── models/                 # Trained ML models and encoders
├── report.py               # Streamlit UI for predictions
├── fjsapo.py               # FJAPSO model training logic
├── requirements.txt        # Dependencies
```

---

## 🚀 Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/your-username/iot-disease-diagnosis.git
cd iot-disease-diagnosis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit App
```bash
streamlit run report.py
```

4. Upload a CSV report and click **"Show Results"** to get disease diagnosis

---

## 📁 Sample Patient Reports
Test the models with:
- `skin_cancer_positive_patient.csv`
- `hiv_positive_patient.csv`
- `healthy_patient_report.csv`

---

## 📌 Future Enhancements
- Add PDF report export
- Expand to time-series analysis with LSTM
- Integrate XAI (e.g., SHAP) for explainable AI

---

## 📄 License
This project is for academic and research purposes under the MIT License.
