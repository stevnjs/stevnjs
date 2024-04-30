import streamlit as st
import pickle
import pandas as pd

# Load the trained model
@st.cache
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Create the Streamlit app
def main():
    st.title('Churn Prediction')

    # User input fields
    st.subheader('Enter Customer Information')
    credit_score = st.number_input('Credit Score', min_value=0, max_value=1000)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=100)
    tenure = st.number_input('Tenure (in years)', min_value=0, max_value=100)
    balance = st.number_input('Account Balance', min_value=0, max_value=100000)
    num_of_products = st.number_input('Number of Products', min_value=1, max_value=10)
    has_credit_card = st.checkbox('Has Credit Card')
    is_active_member = st.checkbox('Is Active Member')
    estimated_salary = st.number_input('Estimated Salary', min_value=0, max_value=1000000)

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_credit_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary]
    })

    # Make predictions
    if st.button('Predict'):
        prediction = model.predict(user_data)
        if prediction[0] == 1:
            st.error('This customer is likely to churn.')
        else:
            st.success('This customer is not likely to churn.')

if __name__ == '__main__':
    main()