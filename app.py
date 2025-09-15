import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("brest_cancer_detection.pickle", "rb"))

# Feature columns
FEATURE_COLUMNS = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry",
    "mean fractal dimension","radius error","texture error","perimeter error",
    "area error","smoothness error","compactness error","concavity error",
    "concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area",
    "worst smoothness","worst compactness","worst concavity",
    "worst concave points","worst symmetry","worst fractal dimension"
]

st.title(" Breast Cancer Prediction App")

# Input method
option = st.radio("Choose input method:", ("Manual Input", "Upload CSV"))

if option == "Manual Input":
    st.subheader("Enter Feature Values Manually")

    input_data = []
    for col_name in FEATURE_COLUMNS:
        value = st.number_input(col_name, min_value=0.0, value=1.0, step=0.01)
        input_data.append(value)

    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.success(f"Prediction: {prediction} (0 = Benign, 1 = Malignant)")
        st.write(f"Benign Probability: {probability[0]:.4f}")
        st.write(f"Malignant Probability: {probability[1]:.4f}")

else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded CSV:", df.head())

        # Check if all required columns exist
        if all(col in df.columns for col in FEATURE_COLUMNS):
            if st.button("Predict from CSV"):
                preds = model.predict(df[FEATURE_COLUMNS])
                df["Prediction"] = preds
                st.write(df.head())
                st.download_button(
                    "Download Results",
                    df.to_csv(index=False),
                    "prediction_results.csv"
                )
        else:
            st.error("Uploaded CSV does not have all required feature columns!")
