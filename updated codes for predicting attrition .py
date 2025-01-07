# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 13:05:00 2024

@author: kufre
"""


import shap

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Load the trained Random Forest model
model_rf = joblib.load(r"C:\Users\kufre\OneDrive\Attrition Prediction Updated\model_rf_smote.sav")

# Define input data for prediction
input_data = (41, 1102, 1, 2, 2, 0, 94, 3, 
              2, 4, 5993, 19479, 8,	1,	11, 3,	1,	0, 8, 0, 1, 6, 4, 0, 5,	0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1
              )

# Convert input data to a NumPy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape input data to match the model's expected input shape
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
predicted_class_prob = model_rf.predict_proba(input_data_reshaped)[0][1]  # Probability for class 1
predicted_class_prob_percent = (predicted_class_prob * 100).round(2)  # Convert to percentage

# Convert probability to binary class (0 or 1)
prediction = model_rf.predict(input_data_reshaped)[0]  # Predict the binary class

# Decision-making logic based on prediction and probability
if prediction == 0:
    if 70 <= predicted_class_prob_percent <= 100:
        print('Negative!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a very low chance of Attrition')
        print('No retention strategy is recommended!')
        
    elif 50 <= predicted_class_prob_percent <= 69:
        print('Negative!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a low chance of Attrition')
        print('No retention strategy is recommended!')
        
    elif 45 <= predicted_class_prob_percent <= 49:
        print('Negative!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a moderate chance of Attrition')
        print('A retention strategy may be recommended!')
    else:
        print('Negative!')
        print('Employee has an uncertain chance of Attrition')
        print('Further analysis may be needed.')

elif prediction == 1:
    if 70 <= predicted_class_prob_percent <= 100:
        print('Positive!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a very high chance of Attrition')
        print('Urgent retention strategy is recommended!')
    elif 59 <= predicted_class_prob_percent <= 69:
        print('Positive!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a high chance of Attrition')
        print('A retention strategy is recommended!')
    elif 45 <= predicted_class_prob_percent <= 58:
        print('Positive!')
        print(f"Predicted Probability: {predicted_class_prob_percent}%")
        print('Employee has a moderate chance of Attrition')
        print('A retention strategy is recommended!')
    else:
        print('Positive!')
        print('Employee has a low chance of Attrition')
        print('Retention may not be necessary.')

import shap

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

X_test = pd.read_csv(r"C:\Users\kufre\OneDrive\Attrition Prediction Updated\X_test.csv")
# Assign proper column names to X_test
X_test.columns = column_names



import shap
import matplotlib.pyplot as plt
import textwrap

# SHAP Explainer
explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(X_test)

# Extract SHAP values for class 1
shap_values_class1 = shap_values[:, :, 1]

# Wrap long column names for better display
wrapped_columns = [textwrap.fill(name, width=10) for name in X_test.columns]  # Adjust width to wrap text

# Replace column names in a copy of X_test for visualization purposes
X_test_wrapped = X_test.copy()
X_test_wrapped.columns = wrapped_columns

# Generate SHAP Force Plot
shap.force_plot(
    base_value=explainer.expected_value[1],
    shap_values=shap_values_class1[0],
    features=X_test_wrapped.iloc[0],  # Pass the first row with wrapped column names
    matplotlib=True  # Force Matplotlib rendering
)

# Adjust figure size to avoid overlapping
fig = plt.gcf()
fig.set_size_inches(25, 12)  # Adjust figure size (width=25, height=12)

plt.tight_layout()  # Ensure content fits within the plot
plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for labels

# Save the plot (optional)
plt.savefig("shap_force_plot_no_overlap.png", bbox_inches='tight', dpi=300)

# Display the plot
plt.show()

