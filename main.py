import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the mapping dictionary for categorical variables
mapping_dict = {
    'Home': {
        'OWN': 0,
        'RENT': 1,
        'MORTGAGE': 2,
        'OTHER': 3
    },
    'Intent': {
        'PERSONAL': 0,
        'EDUCATION': 1,
        'MEDICAL': 2,
        'VENTURE': 3,
        'HOMEIMPROVEMENT': 4,
        'DEBTCONSOLIDATION': 5
    }
}

# Function to map categorical features
def map_categorical_features(data, mappings):
    for column, mapping in mappings.items():
        if column in data.columns:
            try:
                data[column] = data[column].replace(mapping)
            except Exception as e:
                st.error(f"An error occurred when mapping column '{column}': {e}")
        else:
            st.warning(f"The column '{column}' is not found in the DataFrame")
    return data

# Load the saved machine learning model with error handling
@st.cache_resource
def load_model(model_path='rf_risk.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()
if model is None:
    st.stop()  # Stop the app if the model cannot be loaded

# Define the feature columns (excluding 'Default' and 'Id')
feature_columns = ['Age', 'Income', 'Home', 'Emp_length', 'Intent', 
                   'Amount', 'Rate', 'Percent_income', 'Cred_length']

# Sidebar for navigation and contact information
st.sidebar.title("Loan Default Prediction")
st.sidebar.write("Choose an option to input data:")
options = ["Upload CSV", "Manual Input"]
choice = st.sidebar.radio("Select Input Method", options)

# Add contact information to the bottom of the sidebar
st.sidebar.markdown("---")  # Horizontal line for separation
st.sidebar.subheader("Contact Information")
st.sidebar.markdown("""
<div style="font-size: 16px;">
    <table style="width: 100%; border-collapse: collapse;">
        <tr>
            <td><strong>Developer:</strong></td>
            <td>Daniel Borffo Mensah</td>
        </tr>
        <tr>
            <td><strong>Email:</strong></td>
            <td><a href="mailto:borffo.dev@gmail.com">borffo.dev@gmail.com</a></td>
        </tr>
        <tr>
            <td><strong>Phone:</strong></td>
            <td>+233541681348</td>
        </tr>
        <tr>
            <td><strong>LinkedIn:</strong></td>
            <td><a href="https://www.linkedin.com/in/danielborffomensah" target="_blank">LinkedIn Profile</a></td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# Main app content
def main():
    # Add the image at the top of the app using use_container_width
    st.image("https://images.news18.com/ibnlive/uploads/2025/02/the-segment-where-most-of-the-personal-loan-defaults-are-occurring-2025-02-279478183d279e3952b43b24d2232649-16x9.webp", 
             caption="Loan Default Prediction App", 
             use_container_width=True)

    st.title("Loan Default Prediction App")
    st.write("""
    This application predicts the likelihood of a loan default based on the input features.
    You can either upload a CSV file containing the data or manually input the feature values.
    """)

    if choice == "Upload CSV":
        st.header("Upload your CSV file")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.write("### Uploaded Data:")
                st.dataframe(data.head())

                # Preprocess the data
                data_processed = map_categorical_features(data.copy(), mapping_dict)

                # Ensure all feature columns are present
                missing_cols = set(feature_columns) - set(data_processed.columns)
                if missing_cols:
                    st.error(f"The following required columns are missing: {missing_cols}")
                    return

                X = data_processed[feature_columns]

                # Make predictions
                predictions = model.predict(X)
                data['Default_Prediction'] = [
                    "The client is likely to default" if pred == 1 else "The client is unlikely to default" 
                    for pred in predictions
                ]

                st.write("### Prediction Results:")
                st.dataframe(data[['Default_Prediction']].head())

                # Option to download the results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
            except Exception as e:
                st.error(f"Error processing the file: {e}")

    elif choice == "Manual Input":
        st.header("Input Feature Values")

        # Create columns for horizontal layout
        col1, col2, col3 = st.columns(3)

        with col1:
            Age = st.number_input("Age", min_value=18, max_value=100, value=30)
            Home = st.selectbox("Home Status", options=['OWN', 'RENT', 'MORTGAGE', 'OTHER'])
            Intent = st.selectbox("Loan Intent", options=[
                'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'
            ])

        with col2:
            Income = st.number_input("Income", min_value=0, value=50000)
            Emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=150, value=5)
            Amount = st.number_input("Loan Amount", min_value=0, value=10000)

        with col3:
            Rate = st.number_input("Interest Rate", min_value=0.0, max_value=100.0, value=5.5)
            Status = st.selectbox("Credit Score Status", options=['Fully Paid', 'Charged Off'])
            Percent_income = st.number_input("Percentage of Income", min_value=0.0, max_value=100.0, value=20.0)
            Cred_length = st.number_input("Credit History Length (months)", min_value=0, value=24)

        # Add a submit button
        if st.button("Predict"):
            try:
                # Encode Status: Fully Paid -> 1, Charged Off -> 0
                encoded_status = 1 if Status == "Fully Paid" else 0

                input_data = {
                    'Age': Age,
                    'Income': Income,
                    'Home': Home,
                    'Emp_length': Emp_length,
                    'Intent': Intent,
                    'Amount': Amount,
                    'Rate': Rate,
                    'Percent_income': Percent_income,
                    'Cred_length': Cred_length
                }

                input_df = pd.DataFrame([input_data])

                # Map categorical features
                input_df_mapped = map_categorical_features(input_df, mapping_dict)

                # Make prediction
                prediction = model.predict(input_df_mapped)

                # Display the result
                if prediction[0] == 1:
                    st.markdown("<h3 style='color: red;'>Client likely to default.</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: green;'>Client not likely to default.</h3>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ System & Author Info"):
    st.caption("**System:** JBG Logistics")
    st.caption("**Engineer:** Daniel Borffo Mensah  borffo.dev@gmail.com")
    st.caption("**Tech Stack:** Python, Streamlit, Folium, Scikit-Learn")
