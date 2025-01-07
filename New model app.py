# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:24:00 2024

@author: kufre
"""

import numpy as np
import streamlit as st
import joblib
import shap
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define actual column names
column_names = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement',
       'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
       'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
       'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
       'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager', 'BusinessTravel_Non_Travel',
       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
       'Department_Human_Resources', 'Department_Research_and_Development',
       'Department_Sales', 'JobRole_Healthcare_Representative',
       'JobRole_Human_Resources', 'JobRole_Laboratory_Technician',
       'JobRole_Manager', 'JobRole_Manufacturing_Director',
       'JobRole_Research_Director', 'JobRole_Research_Scientist',
       'JobRole_Sales_Executive', 'JobRole_Sales_Representative',
       'MaritalStatus_Divorced', 'MaritalStatus_Married',
       'MaritalStatus_Single']

# Load the saved model
loaded_model = joblib.load(open(r"C:\Users\kufre\OneDrive\Attrition Prediction Updated\model_rf_smote.sav", 'rb'))
X_test = pd.read_csv(r"C:\Users\kufre\OneDrive\Attrition Prediction Updated\X_test.csv")
# Assign proper column names to X_test
X_test.columns = column_names

def Employee_Attrition_Prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    predicted_class_prob = loaded_model.predict_proba(input_data_reshaped)[0][1]
    predicted_class_prob_percent = (predicted_class_prob * 100).round(2)
    prediction = loaded_model.predict(input_data_reshaped)[0]
    
    # Initialize message
    message = ""

    # Prediction Logic
    if prediction == 0:
        if 70 <= predicted_class_prob_percent <= 100:
            message = (
                "Negative!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a very low chance of Attrition.\n"
                "\n"
                "No retention strategy is recommended!"
            )
        elif 50 <= predicted_class_prob_percent <= 69:
            message = (
                "Negative!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a low chance of Attrition.\n"
                "\n"
                "No retention strategy is recommended!"
            )
        elif 45 <= predicted_class_prob_percent <= 49:
            message = (
                "Negative!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a moderate chance of Attrition.\n"
                "\n"
                "A retention strategy may be recommended!"
            )
        else:
            message = (
                "Negative!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has an uncertain chance of Attrition.\n"
                "\n"
                "Further analysis may be needed."
            )
    elif prediction == 1:
        if 70 <= predicted_class_prob_percent <= 100:
            message = (
                "Positive!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a very high chance of Attrition.\n"
                "\n"
                "Urgent retention strategy is recommended!"
            )
        elif 59 <= predicted_class_prob_percent <= 69:
            message = (
                "Positive!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a high chance of Attrition.\n"
                "\n"
                "A retention strategy is recommended!"
            )
        elif 45 <= predicted_class_prob_percent <= 58:
            message = (
                "Positive!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a moderate chance of Attrition.\n"
                "\n"
                "A retention strategy is recommended!"
            )
        else:
            message = (
                "Positive!\n"
                "\n"
                f"Predicted Probability: {predicted_class_prob_percent}%\n"
                "\n"
                "Employee has a low chance of Attrition.\n"
                "\n"
                "Retention may not be necessary."
            )
    
    # Print the message
    print(message)

    # SHAP Force Plot Logic
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X_test)

    # Extract SHAP values for class 1
    shap_values_class1 = shap_values[:, :, 1]

    # Wrap long column names for better display
    wrapped_columns = [textwrap.fill(name, width=10) for name in X_test.columns]
    X_test_wrapped = X_test.copy()
    X_test_wrapped.columns = wrapped_columns

    # Generate SHAP Force Plot
    shap.force_plot(
        base_value=explainer.expected_value[1],
        shap_values=shap_values_class1[0],
        features=X_test_wrapped.iloc[0],
        matplotlib=True
    )

    # Adjust figure size to avoid overlapping
    fig = plt.gcf()
    fig.set_size_inches(25, 12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    # Render the plot in Streamlit
    st.pyplot(plt.gcf())  # Capture the current figure and render it in Streamlit
    

    # Return the text message
    return message

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 1

# Function to navigate pages
def next_page():
    if st.session_state.page < 11:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

# Main function
def main():
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        .input-container input {
            font-size: 16px;
            padding: 5px;
        }
        .image-container img {
            width: 100%;
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Employee Attrition Prediction Web App")

    # Two-column layout for each page
    left_column, right_column = st.columns([1, 1.5])

    # Page-specific content
    if st.session_state.page == 1:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\HR image front page.jpg", use_container_width=True)
        with left_column:
            st.header("Page 1: Employee Information")
            st.text_input('Employee Age: (Range: 18 - 60)', key='Age')
            st.text_input('Employee Daily Rate: (Range: 50 - 2000)', key='DailyRate')
            st.text_input('Employee Distance From Home: (Range: 1 - 29)', key='DistanceFromHome')
            st.text_input('Employee Education Level: (Range: 1 - 5)', key='Education')

    elif st.session_state.page == 2:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\salary.jpeg", use_container_width=True)
        with left_column:
            st.header("Page 2: Employee Information")
            st.text_input('Work Environment Satisfaction Level: (Range: 1 - 4)', key='EnvironmentSatisfaction')
            st.text_input('Gender: Male(1), Female(0)', key='Gender')
            st.text_input('Employee Hourly Rate: (Range: 30 - 100)', key='HourlyRate')
            st.text_input('Job Involvement Level: (Range: 1 - 4)', key='JobInvolvement')
           

    elif st.session_state.page == 3:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\page 3.jpg", use_container_width=True)
        with left_column:
            st.header("Page 3: Employee Information")
            st.text_input('Employee Job Level: (Range: 1 - 5)', key='JobLevel')
            st.text_input('Job Satisfaction Level: (Range: 1 - 4)', key='JobSatisfaction')
            st.text_input('Employee Monthly Income: (Range: 1000 - 19999)', key='MonthlyIncome')
            st.text_input('Employee Monthly Rate: (Range: 2000 - 27000)', key='MonthlyRate')

            

    elif st.session_state.page == 4:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\round table.jpg", use_container_width=True)
        with left_column:
            st.header("Page 4: Employee Information")
            st.text_input('Number of Companies Worked For: (Range: 0 - 9)', key='NumCompaniesWorked')
            st.text_input('Does Employee Work Over Time?: Yes(1), No(0)', key='OverTime')
            st.text_input('Percentage Salary Hike: (Range: 11 - 25)', key='PercentSalaryHike')
            st.text_input('Employee Performance Rating: (Range: 1 - 5)', key='PerformanceRating')

            
    elif st.session_state.page == 5:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\work life balance.jpg", use_container_width=True)
        with left_column:
            st.header("Page 5: Employee Information")
            st.text_input('Relationship Satisfaction: (Range: 1 - 4)', key='RelationshipSatisfaction')
            st.text_input('Stock Option Level: (Range: 0 - 3)', key='StockOptionLevel')
            st.text_input('Total Working Years: (Range: 0 - 40)', key='TotalWorkingYears')
            st.text_input('Training Times Last Year: (Range: 0 - 6)', key='TrainingTimesLastYear')


    elif st.session_state.page == 6:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\promotion picture.webp", use_container_width=True)
        with left_column:
            st.header("Page 6: Employee Information")
            st.text_input('Work Life Balance: (Range: 1 - 4)', key='WorkLifeBalance')
            st.text_input('Years at Company: (Range: 0 - 40)', key='YearsAtCompany')
            st.text_input('Employee Years In Current Role: (Range: 0 - 40)', key='YearsInCurrentRole')
            st.text_input('Employee Years Since Last Promotion: (Range: 0 - 15)', key='YearsSinceLastPromotion')


    elif st.session_state.page == 7:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\business trip.jpeg", use_container_width=True)
        with left_column:
            st.header("Page 7: Employee Information")
            st.text_input('Employee Years With Current Manager: (Range: 0 - 17)', key='YearsWithCurrManager')
            st.text_input('Business Non-Travel Requirement: None(1), Yes(0)', key='BusinessTravel_Non_Travel')
            st.text_input('Does Employee Frequently Travel?: Yes(1), No(0)', key='BusinessTravel_Travel_Frequently')
            st.text_input('Does Employee Rarely Travel?: Yes(1), No(0)', key='BusinessTravel_Travel_Rarely')


    elif st.session_state.page == 8:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\customer care.jpg", use_container_width=True)
        with left_column:
            st.header("Page 8: Employee Information")
            st.text_input('Is Employee in Human Resources Department?: Yes(1), No(0)', key='Department_Human_Resources')
            st.text_input('Is Employee in Research & Development Department?: Yes(1), No(0)', key='Department_Research_and_Development')
            st.text_input('Is Employee in Sales Department?: Yes(1), No(0)', key='Department_Sales')
            st.text_input('Is Employee a Healthcare Representative?: Yes(1), No(0)', key='JobRole_Healthcare_Representative')

    elif st.session_state.page == 9:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\manufacturing picture.jpg", use_container_width=True)
        with left_column:
            st.header("Page 9: Employee Information")
            st.text_input('Is Employee job role in Human Resources?: Yes(1), No(0)', key='JobRole_Human_Resources')
            st.text_input('Is Employee a Laboratory Technician?: Yes(1), No(0)', key='JobRole_Laboratory_Technician')
            st.text_input('Is Employee a Manager?: Yes(1), No(0)', key='JobRole_Manager')
            st.text_input('Is Employee a Manufacturing Director?: Yes(1), No(0)', key='JobRole_Manufacturing_Director')




    elif st.session_state.page == 10:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\lab tech picture.jpg", use_container_width=True)
        with left_column:
            st.header("Page 10: Employee Information")
            st.text_input('Is Employee a Research Director?: Yes(1), No(0)', key='JobRole_Research_Director')
            st.text_input('Is Employee a Research Scientist?: Yes(1), No(0)', key='JobRole_Research_Scientist')
            st.text_input('Is Employee a Sales Executive?: Yes(1), No(0)', key='JobRole_Sales_Executive')
            st.text_input('Is Employee a Sales Representative?: Yes(1), No(0)', key='JobRole_Sales_Representative')
            

    elif st.session_state.page == 11:
        with right_column:
            st.image(r"C:\Users\kufre\OneDrive\Desktop\Attrition Prediction model\HR image 1.jpeg", use_container_width=True)
        with left_column:
            st.header("Page 11: Employee Information")
            st.text_input('Is Employee Divorced?: Yes(1), No(0)', key='MaritalStatus_Divorced')
            st.text_input('Is Employee Married?: Yes(1), No(0)', key='MaritalStatus_Married')
            st.text_input('Is Employee Single?: Yes(1), No(0)', key='MaritalStatus_Single')
           

            if st.button('Predict'):
                inputs = [
                    float(st.session_state.get('Age', 0)),
                    float(st.session_state.get('DailyRate', 0)),
                    float(st.session_state.get('DistanceFromHome', 0)),
                    float(st.session_state.get('Education', 0)),
                    float(st.session_state.get('EnvironmentSatisfaction', 0)),
                    float(st.session_state.get('Gender', 0)),
                    float(st.session_state.get('HourlyRate', 0)),
                    float(st.session_state.get('JobInvolvement', 0)),
                    float(st.session_state.get('JobLevel', 0)),
                    float(st.session_state.get('JobSatisfaction', 0)),
                    float(st.session_state.get('MonthlyIncome', 0)),
                    float(st.session_state.get('MonthlyRate', 0)),
                    float(st.session_state.get('NumCompaniesWorked', 0)),
                    float(st.session_state.get('OverTime', 0)),
                    float(st.session_state.get('PercentSalaryHike', 0)),
                    float(st.session_state.get('PerformanceRating', 0)),
                    float(st.session_state.get('RelationshipSatisfaction', 0)),
                    float(st.session_state.get('StockOptionLevel', 0)),
                    float(st.session_state.get('TotalWorkingYears', 0)),
                    float(st.session_state.get('TrainingTimesLastYear', 0)),
                    float(st.session_state.get('WorkLifeBalance', 0)),
                    float(st.session_state.get('YearsAtCompany', 0)),
                    float(st.session_state.get('YearsInCurrentRole', 0)),
                    float(st.session_state.get('YearsSinceLastPromotion', 0)),
                    float(st.session_state.get('YearsWithCurrManager', 0)),
                    float(st.session_state.get('BusinessTravel_Non_Travel', 0)),
                    float(st.session_state.get('BusinessTravel_Travel_Frequently', 0)),
                    float(st.session_state.get('BusinessTravel_Travel_Rarely', 0)),
                    float(st.session_state.get('Department_Human_Resources', 0)),
                    float(st.session_state.get('Department_Research_and_Development', 0)),
                    float(st.session_state.get('Department_Sales', 0)),
                    float(st.session_state.get('JobRole_Healthcare_Representative', 0)),
                    float(st.session_state.get('JobRole_Human_Resources', 0)),
                    float(st.session_state.get('JobRole_Laboratory_Technician', 0)),
                    float(st.session_state.get('JobRole_Manager', 0)),
                    float(st.session_state.get('JobRole_Manufacturing_Director', 0)),
                    float(st.session_state.get('JobRole_Research_Director', 0)),
                    float(st.session_state.get('JobRole_Research_Scientist', 0)),
                    float(st.session_state.get('JobRole_Sales_Executive', 0)),
                    float(st.session_state.get('JobRole_Sales_Representative', 0)),
                    float(st.session_state.get('MaritalStatus_Divorced', 0)),
                    float(st.session_state.get('MaritalStatus_Married', 0)),
                    float(st.session_state.get('MaritalStatus_Single', 0)),

                ]
                Attrition_Status = Employee_Attrition_Prediction(inputs)
                st.success(f'Prediction: {Attrition_Status}')
                
                

    # Navigation buttons under image in the right column
    with right_column:
        if st.session_state.page > 1:
            st.button("Previous", on_click=prev_page)
        if st.session_state.page < 11:
            st.button("Next", on_click=next_page)

if __name__ == '__main__':
    main()
