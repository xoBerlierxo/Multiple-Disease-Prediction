# 🩺 Multiple-Disease-Prediction

An interactive machine learning-powered web application that can predict the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** based on user-provided medical information.

Built using **Streamlit**, this app integrates three pre-trained SVM classifiers and provides a clean, user-friendly interface with real-time predictions.

## 🚀 Features

- 🎯 Predicts three medical conditions:
  - Diabetes
  - Heart Disease
  - Parkinson's Disease
- 🧠 Machine Learning Models trained using **Support Vector Machines (SVM)**
- 💾 Models serialized using **Pickle**
- 🖥️ Interactive web interface built with **Streamlit**
- 🔘 Uses Streamlit's **option menu** for multi-page navigation
- ✅ Real-time user input and prediction with intuitive UX

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python, SVM models (scikit-learn)
- **Model Serialization**: Pickle
- **Environments**: Anaconda

---

## 📂 Project Structure
- models
- - heart_model.sav
  - diabetes_model.sav
  - parkinsons_model.pkl
- Multiple_disease_prediction.py
- requirements.txt

## 🧪 How to Run Locally

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app
```

2. **Create and Activate Environment**

```bash
conda create -n disease-predictor python=3.10
conda activate disease-predictor
```
3. **Install Dependencies**

```bash
pip install -r requirements.txt
Run the Streamlit App
```
4. **Run The python File**
```bash
streamlit run app.py
```
