# This a python written web application that predicts multiple diseases based on user input and machine learning models.
import streamlit as st
from streamlit_option_menu import option_menu
import pickle

# Loading the saved models 
diabetes_model = pickle.load(open("C:/Users/somsh/Jupyter Notebooks/ML/Streamlit/Models/diabetes_model.sav", 'rb'))
heart_model = pickle.load(open("C:/Users/somsh/Jupyter Notebooks/ML/Streamlit/Models/heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open("C:/Users/somsh/Jupyter Notebooks/ML/Streamlit/Models/parkinsons_model.pkl", 'rb'))

# using the sidebar to navigate
with st.sidebar:
    selected = option_menu(
        'Multiple Disease prediction System',
        [
            'Diabetes Prediction System',
            'Heart Disease Prediction System',
            'Parkinsons Disease Prediciton System'
        ],
        icons = ['activity', 'heart', 'person'],
        default_index = 0
        )

# Now implementing the each of the individual components
# Diabetes Prediction
if selected == 'Diabetes Prediction System' :
    st.title("Diabetes Prediction System using SVM")

    # Getting the input from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age')
    
    # creating a button for prediction
    if st.button("Diabetes Test result Prediction") :
        diabetes_prediction_= diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,Age]])
        
        if diabetes_prediction_ == [0] :
            st.success("The Person is not Diabetic")
        
        if diabetes_prediction_ == [1] :
            st.success("The Person is Diabetic")

# Heart Disease Prediction
if selected == 'Heart Disease Prediction System' :
    st.title("Heart Disease Prediction System using SVM")

    # Getting the input from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain Type (cp)')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure (trestbps)')

    with col2:
        chol = st.text_input('Cholesterol (chol)')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (fbs)')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results (restecg)')

    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved (thalach)')

    with col3:
        exang = st.text_input('Exercise Induced Angina (exang)')

    with col1:
        oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest (oldpeak)')

    with col2:
        slope = st.text_input('Slope of the Peak Exercise ST Segment (slope)')

    with col3:
        ca = st.text_input('Number of Major Vessels Colored by Fluoroscopy (ca)')

    with col1:
        thal = st.text_input('Thal (thal)')
    
    # creating a button for prediction
    if st.button("Heart Disease Test result Prediction"):
        heart_prediction_ = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if heart_prediction_ == [0]:
        st.success("The Person does NOT have Heart Disease")

    elif heart_prediction_ == [1]:
        st.success("The Person has Heart Disease")


# Parkinson's Disease prediction
if selected == 'Parkinsons Disease Prediciton System' :
    st.title("Parkinson's Disease Prediction System using SVM")

    # Getting the input from the user for Parkinson's prediction
    col1, col2, col3 = st.columns(3)

    with col1:
        # MDVP:Fo(Hz)
        fo_hz = st.text_input('MDVP:Fo(Hz)')
    with col2:
        # MDVP:Fhi(Hz)
        fhi_hz = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        # MDVP:Flo(Hz)
        flo_hz = st.text_input('MDVP:Flo(Hz)')

    with col1:
        # MDVP:Jitter(%)
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col2:
        # MDVP:Jitter(Abs)
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
        # MDVP:RAP
        rap = st.text_input('MDVP:RAP')

    with col1:
        # MDVP:PPQ
        ppq = st.text_input('MDVP:PPQ')
    with col2:
        # Jitter:DDP
        jitter_ddp = st.text_input('Jitter:DDP')
    with col3:
        # MDVP:Shimmer
        shimmer = st.text_input('MDVP:Shimmer')

    with col1:
        # MDVP:Shimmer(dB)
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')
    with col2:
        # Shimmer:APQ3
        shimmer_apq3 = st.text_input('Shimmer:APQ3')
    with col3:
        # Shimmer:APQ5
        shimmer_apq5 = st.text_input('Shimmer:APQ5')

    with col1:
        # MDVP:APQ
        mdvp_apq = st.text_input('MDVP:APQ')
    with col2:
        # Shimmer:DDA
        shimmer_dda = st.text_input('Shimmer:DDA')
    with col3:
        # NHR
        nhr = st.text_input('NHR')

    with col1:
        # HNR
        hnr = st.text_input('HNR')
    with col2:
        # RPDE
        rpde = st.text_input('RPDE')
    with col3:
        # DFA
        dfa = st.text_input('DFA')

    with col1:
        # spread1
        spread1 = st.text_input('spread1')
    with col2:
        # spread2
        spread2 = st.text_input('spread2')
    with col3:
        # D2
        d2 = st.text_input('D2')

    with col1: # Using col1 for the last feature as it's the 22nd feature
        # PPE
        ppe = st.text_input('PPE')


    # creating a button for prediction
    if st.button("Parkinson's Test Result Prediction"):
        # Convert all inputs to float and gather them into a list
        # The 'name' column from your array is typically an identifier and not used in the model,
        # so it's excluded from the input features.
        try:
            parkinsons_input_features = [
                float(fo_hz), float(fhi_hz), float(flo_hz),
                float(jitter_percent), float(jitter_abs), float(rap), float(ppq), float(jitter_ddp),
                float(shimmer), float(shimmer_db), float(shimmer_apq3), float(shimmer_apq5),
                float(mdvp_apq), float(shimmer_dda), float(nhr), float(hnr),
                float(rpde), float(dfa), float(spread1), float(spread2), float(d2), float(ppe)
            ]

            # Assuming 'parkinsons_model' is your pre-trained model
            # Make sure the order of features here matches the order your model was trained on!
            parkinsons_prediction_ = parkinsons_model.predict([parkinsons_input_features])

            if parkinsons_prediction_ == [0]:
                st.success("The Person is NOT predicted to have Parkinson's disease.")
            elif parkinsons_prediction_ == [1]:
                st.success("The Person IS predicted to have Parkinson's disease.")
            else:
                st.warning("Prediction result is unexpected.")

        except ValueError:
            st.error("Please ensure all input fields contain valid numerical values.")
        except NameError:
            st.error("Error: 'parkinsons_model' is not defined. Make sure your model is loaded.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

