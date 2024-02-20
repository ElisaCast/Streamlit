# Local URL: http://localhost:8501
# Network URL: http://192.168.8.187:8501

from pickle import load
import streamlit as st
import pandas as pd
try:
    import streamlit as st
    # Resto del código de tu aplicación Streamlit
except ImportError:
    print("Streamlit no está instalado. Por favor, instálalo utilizando 'pip install streamlit'.")


model = load(open("C:/Users/elisa/OneDrive/Escritorio/Data Sciencie/ML-BosstingAlgoritmo/models/boosting_classifier_nestimators-20_learnrate-0.001_42.sav", "rb"))

class_dict = {
    "0": "No tiene diabetes",
    "1": "Tiene diabetes",
}

# Titulo de la aplicación
st.title("Diabetes - Model prediction")

# Deslizadores para las características
age = st.slider("Age", min_value=0, max_value=120, step=1)
pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.slider("Glucose", min_value=0, max_value=200, step=1)
blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=150, step=1)
bmi = st.slider("BMI", min_value=0.0, max_value=60.0, step=0.1)
diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, step=0.01)
insulin = st.slider("Insulin", min_value=0.0, max_value=500.0, step=1.0)  
skin_thickness = st.slider("Skin Thickness", min_value=0.0, max_value=100.0, step=1.0)  

# Botón para realizar la predicción
if st.button("Predict Diabetes"):

# Crear un DataFrame con las características ingresadas
    input_data = pd.DataFrame({
        'Insulin': [insulin], 
        'Age': [age],
        'Pregnancies': [pregnancies],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'BMI': [bmi],
        'BloodPressure': [blood_pressure],
        'Glucose': [glucose],
        'SkinThickness': [skin_thickness] 
    })

# Realizar la predicción usando el modelo
    prediction = model.predict(input_data)[0]  # Aquí obtenemos el primer elemento del array resultante

 # Convertir la predicción a 1 o 0
    binary_prediction = 1 if prediction >= 0.5 else 0

# Obtener la etiqueta asociada a la predicción
    predicted_label = class_dict[str(binary_prediction)]

# Mostrar la predicción
    st.write("Prediction:", predicted_label)
    
