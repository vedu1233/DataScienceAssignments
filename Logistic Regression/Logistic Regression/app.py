import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = pickle.load(open('titanic_model.pkl', 'rb'))

# Load the scaler used during training
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit app
def main():
    # Add custom CSS to include a background image
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("titanic12.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('Survival Prediction on Titanic')

    # User inputs for the 7 features
    Pclass = st.selectbox('Passenger Class (Pclass)', ['1st Class', '2nd Class', '3rd Class'])
    Pclass = [1, 2, 3][['1st Class', '2nd Class', '3rd Class'].index(Pclass)]  # Convert to numeric

    Sex = st.selectbox('Sex', ['Female', 'Male'])
    Sex = 1 if Sex == 'Male' else 0  # Convert to numeric

    Age = st.number_input('Age', min_value=0, max_value=100, value=25)
    Fare = st.number_input('Fare', min_value=0, value=50)
    
    # Embarked options, converted to one-hot encoding
    Embarked = st.selectbox('Embarked Port', ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)'])
    Embarked_C = 1 if Embarked == 'Cherbourg (C)' else 0
    Embarked_Q = 1 if Embarked == 'Queenstown (Q)' else 0
    Embarked_S = 1 if Embarked == 'Southampton (S)' else 0

    # Create a NumPy array for input features
    input_data = np.array([[Pclass, Sex, Age, Fare, Embarked_C, Embarked_Q, Embarked_S]])

    # Scale the input data using the same scaler as during training
    scaled_data = scaler.transform(input_data)

    # Predict using the trained model
    if st.button('Predict'):
        prediction = model.predict(scaled_data)
        if prediction[0] == 1:
            st.success('The passenger is predicted to have survived.')
        else:
            st.warning('The passenger is predicted to not have survived.')

if __name__ == '__main__':
    main()
