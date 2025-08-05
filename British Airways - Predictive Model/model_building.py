#!/usr/bin/env python
# coding: utf-8

# # British Airways: Predicting Customer Buying Behaviour – Summary
# 
# This notebook presents a complete machine learning workflow for predicting customer purchase behavior using the British Airways dataset. The primary goal is to identify potential buyers from a large pool of users to support targeted marketing and improve conversion rates.
# 
# We implemented a robust pipeline combining preprocessing (scaling and encoding), class balancing using SMOTE, and a Random Forest Classifier. Evaluation metrics (accuracy, precision, recall, F1-score, and confusion matrix) were analyzed on the test set to assess model performance.
# 
# While the model achieves high accuracy (82.5%) and performs well on the majority class (non-buyers), precision and recall for the minority class (buyers) indicate room for improvement. Further enhancements can include trying alternative models, threshold tuning, and more feature engineering.
# 
# This project is part of the British Airways Virtual Experience and showcases practical data science techniques applied to a real-world business scenario.
# 

# ### Understanding Dataset

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Load the dataset
df = pd.read_csv("customer_booking.csv" , encoding="ISO-8859-1")


# **Initial Data Inspection: Check the shape and preview the first few rows to understand the structure of the dataset.**

# In[3]:


print(df.shape)


# In[4]:


df.head()


# In[5]:


df.info()


# **Checking for Missing Values: Identify any missing values in the dataset that may need to be handled before modeling.**

# In[6]:


print(df.isnull().sum())


# ### Data Cleaning & Exploration

# In[7]:


df.duplicated().sum()


# In[8]:


print(df.describe(include='all'))


# In[9]:


# Check current data types
print(df.dtypes)


# #### Exploring Categorical Feature Distributions
# 
# To understand the frequency distribution of key categorical features, we print the value counts for:
# 
# - `sales_channel`: Distribution of bookings through different sales channels (e.g., online vs offline).
# - `trip_type`: Types of trips (e.g., one-way, round-trip).
# - `flight_day`: Which day of the week flights are scheduled.
# - `route`: Popular flight routes taken by passengers.
# - `booking_origin`: Countries or regions where bookings originated.
# 
# This helps identify potential class imbalances and business insights.

# In[10]:


# Step 2.4: Categorical Value Counts
for col in ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']:
    print(f"\n{col}:\n", df[col].value_counts())


# #### Numeric Feature Distributions
# Visualizing the distribution of key numeric features to detect skewness, outliers, or patterns that may impact model performance.

# In[11]:


# Step 2.5: Distribution Plots (Numeric)
import seaborn as sns
import matplotlib.pyplot as plt

for col in ['num_passengers', 'purchase_lead', 'length_of_stay', 'flight_hour', 'flight_duration']:
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(col)
    plt.show()


# #### Categorical Feature Counts
# Displayed count plots for selected categorical features to understand their frequency distribution and class imbalance.

# In[12]:


# Step 2.6: Count Plots (Categorical)
for col in ['sales_channel', 'trip_type', 'flight_day']:
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()


# #### Correlation Heatmap
# Visualizing correlations between numeric features to identify potential multicollinearity and key relationships.

# In[13]:


# Step 2.7: Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# #### Remove Duplicate Rows
# Eliminated any duplicate records from the dataset to ensure data integrity and prevent bias in analysis.

# In[14]:


# Remove duplicate rows
df = df.drop_duplicates()


# In[15]:


# Confirm removal
print("New shape after removing duplicates:", df.shape)


# ### Creating a preprocessing pipeline using ColumnTransformer + Pipeline.

# ##### Define Numeric and Categorical Columns
# 
# Identified and separated numerical and categorical columns to streamline preprocessing.  
# - `numeric_cols` include continuous variables such as `length_of_stay`, `purchase_lead`, and `flight_hour`.  
# - `categorical_cols` include non-numeric attributes such as `sales_channel`, `trip_type`, `flight_day`, `route`, and `booking_origin`.
# 

# In[16]:


# Step 1: Define Numeric and Categorical Columns
numeric_cols = ['length_of_stay', 'purchase_lead', 'flight_hour']
categorical_cols = ['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']


# ##### Create ColumnTransformer
# 
# Constructed a `ColumnTransformer` to preprocess different column types in a pipeline.  
# - **Numeric columns** are scaled using `StandardScaler`.  
# - **Categorical columns** are encoded using `OneHotEncoder` with `handle_unknown='ignore'` to manage unseen categories during prediction.
# 

# In[17]:


# Step 2: Create ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


# ##### Set Up Full Pipeline with Classifier
# 
# Created a complete `Pipeline` combining preprocessing and modeling steps:  
# - First step: `preprocessing` applies transformations defined in the `ColumnTransformer`.  
# - Second step: a `RandomForestClassifier` is used as the placeholder model with a fixed `random_state` for reproducibility.
# 

# In[18]:


# Step 3: Set Up Full Pipeline with a Classifier (placeholder)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])


# ## Model Building

# ##### Split the Data
# 
# Split the dataset into training and testing sets using `train_test_split` with:
# - 80% for training and 20% for testing.
# - `stratify=y` ensures class distribution is preserved across both sets.
# - `random_state=42` is used for reproducibility.
# 

# In[19]:


# Split the data
from sklearn.model_selection import train_test_split

X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ##### Check Target Class Distribution
# 
# Check the distribution of the target variable `booking_complete` using normalized value counts.  
# This reveals the proportion of each class (0 and 1) to understand class imbalance, which is important for model evaluation.
# 

# In[20]:


y.value_counts(normalize=True)


# ##### Handle Class Imbalance with SMOTE
# 
# First, encode `X_train` using the defined `preprocessor` to transform numerical and categorical features.  
# Then, apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the classes in the training data by generating synthetic examples of the minority class. This helps improve model performance on imbalanced datasets.
# 

# In[22]:


# Step: Encode X_train before applying SMOTE
X_train_encoded = preprocessor.fit_transform(X_train)

# Step: Apply SMOTE on encoded training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)


# ##### Apply SMOTE to Balance Training Classes
# 
# Use SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.  
# This ensures a balanced training dataset, which helps the model learn from both classes more effectively.
# 

# ##### Import SMOTE-Compatible Pipeline
# 
# Import the `Pipeline` class from `imblearn.pipeline` to enable seamless integration of SMOTE with preprocessing and model training steps.
# 

# In[23]:


# Step 1: Import the right pipeline class for SMOTE integration
from imblearn.pipeline import Pipeline


# ##### Build Full Pipeline with Preprocessing, SMOTE, and Classifier
# 
# Create a complete pipeline that includes preprocessing (scaling and encoding), SMOTE to handle class imbalance, and a Random Forest Classifier for prediction.
# 

# In[28]:


# Build the full pipeline with preprocessing, SMOTE, and classifier
# Define model_pipeline ONCE using ImbPipeline
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor), # Your ColumnTransformer
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])


# ## Model Tuning

# ##### Cross-Validation with Full Pipeline
# 
# Perform 5-fold cross-validation using the complete pipeline. Evaluate model performance using the F1 score to account for class imbalance.
# 

# In[29]:


# Step 3: Cross-validation with pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

scorer = make_scorer(f1_score)
scores = cross_val_score(model_pipeline, X_train, y_train, scoring=scorer, cv=5)

print("Cross-validated F1 scores:", scores)
print("Mean F1 Score:", scores.mean())


# #### Cross-Validation Results
# 
# The cross-validated F1 scores across 5 folds are:
# 
# - Fold 1: 0.3225  
# - Fold 2: 0.2735  
# - Fold 3: 0.3261  
# - Fold 4: 0.3002  
# - Fold 5: 0.2925  
# 
# **Mean F1 Score:** 0.303
# 
# This performance indicates that the model is moderately effective at handling the class imbalance, but there may still be room for improvement via hyperparameter tuning or using more complex models.
# 

# #### Train Classifier on Resampled Data
# 
# Extract the classifier from the pipeline and train it directly on the SMOTE-resampled and encoded training data.
# 

# In[30]:


# Extract classifier from pipeline
clf = pipeline.named_steps['clf']

# Fit the classifier directly on the resampled (already encoded) data
clf.fit(X_train_resampled, y_train_resampled)


# #### Fit and Predict on Resampled Data (Manual SMOTE Approach)
# 
# To better handle class imbalance and improve generalization:
# 
# 1. **Encode Training Data**: The `preprocessor` (with `StandardScaler` and `OneHotEncoder`) was fitted to `X_train` and applied to transform it into a numeric format.
# 2. **SMOTE Oversampling**: The `SMOTE` algorithm was applied to synthetically generate new examples for the minority class in the training data, resulting in a balanced dataset (`X_resampled`, `y_resampled`).
# 3. **Model Training**: A `RandomForestClassifier` with 100 trees and a fixed random state was trained on the resampled data.
# 4. **Test Data Transformation & Prediction**: The test set was transformed using the same fitted `preprocessor`, and predictions (`y_pred`) were made on it using the trained model.
# 
# This approach allows the model to learn from a balanced dataset while maintaining consistency in feature preprocessing.
# 

# In[32]:


# Step 1: Encode X_train using the preprocessor
X_train_encoded = preprocessor.fit_transform(X_train)

# Step 2: Apply SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_encoded, y_train)

# Step 3: Define and fit model on resampled data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Step 4: Transform X_test and predict
X_test_encoded = preprocessor.transform(X_test)
y_pred = model.predict(X_test_encoded)


# ## Model Evaluation

# In[33]:


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# STEP 1: Print evaluation metrics (no confusion matrix yet)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# #### Key Findings from Model Evaluation
# 
# - **High Overall Accuracy (82.5%)**, but it is **misleading** due to class imbalance.
# - The model performs **very well on class 0** (majority class) with:
#   - Precision: 0.87
#   - Recall: 0.93
#   - F1 Score: 0.90
# - **Weak performance on class 1** (minority class):
#   - Precision: 0.37 → High false positives
#   - Recall: 0.25 → Many false negatives
#   - F1 Score: 0.30 → Poor balance between precision and recall
# - **Macro average F1 score is only 0.60**, confirming the imbalance issue.
# - **Weighted average** favors class 0, inflating overall metrics.
# 
# #### Insight
# > The model is biased toward the majority class and is not reliable for identifying minority class instances. Remedial steps such as **SMOTE**, **threshold adjustment**, or **balanced class weights** are needed to improve minority class recall and F1 score.
# 

# In[34]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Greens', values_format='d')
plt.title("Confusion Matrix")
plt.show()


# #### Confusion Matrix Insights
# 
# |               | Predicted: 0 | Predicted: 1 |
# |---------------|--------------|--------------|
# | Actual: 0     |     7783     |     596      |
# | Actual: 1     |     1113     |     365      |
# 
# #### Observations:
# 
# - **True Negatives (TN)**: 7783 → Model correctly predicted most of the class 0 instances.
# - **False Positives (FP)**: 596 → Model mistakenly predicted class 1 for many actual class 0 cases.
# - **False Negatives (FN)**: 1113 → Model failed to identify a large number of class 1 cases.
# - **True Positives (TP)**: 365 → Correct predictions for class 1 are low.
# 
# #### Key Insight:
# 
# > The model is **highly skewed towards class 0**, leading to a large number of **false negatives**. This is critical if class 1 represents an important outcome (e.g., churn, fraud, disease), as many positive cases are being missed.
# >
# > Addressing class imbalance and improving recall for class 1 is essential for a more reliable model.
# 

# ### Trying fitting a Random Forest model with the same SMOTE-resampled data and compare the performance

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Step 1: Initialize and train Random Forest on SMOTE-resampled data
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Step 2: Transform X_test using the original preprocessor
X_test_encoded = preprocessor.transform(X_test)

# Step 3: Predict on X_test
y_pred_rf = rf_model.predict(X_test_encoded)

# Step 4: Evaluate performance (excluding confusion matrix for now)
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

print("Random Forest Evaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# Overall Performance
# Accuracy is 82.5%, which looks high — but this is due to class imbalance.
# 
# The model predicts class 0 (non-buyers, majority class) very well, but struggles with class 1 (buyers).
# 
# Class Imbalance Effects
# Precision for buyers (class 1): 0.3736
# → When the model predicts a buyer, it’s correct ~37% of the time.
# 
# Recall for buyers (class 1): 0.2470
# → It only finds ~25% of actual buyers.
# 
# F1 Score (class 1): 0.2974
# → Overall weak performance on minority class.
# 
# Conclusion
# Even though the Random Forest performs slightly better than a dummy model, it still fails to capture the minority class well — a common problem with imbalanced datasets.
# 

# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Step 1: Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 2: Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()


# Issue
# Your model performs very well on class 0 (non-target class) but struggles with class 1 (target class):
# 
# Recall for class 1 is only 25%, which means many actual positives are being missed.
# 
# Precision for class 1 is ~37%, meaning many predicted positives are incorrect.

# In[37]:


# Recalling pipelines names

print(pipeline.named_steps)


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the preprocessor from the pipeline
preprocessor = pipeline.named_steps['preprocessing']

# Extract numerical and categorical feature names
num_features = preprocessor.transformers_[0][2]  # ['length_of_stay', 'purchase_lead', 'flight_hour']
cat_encoder = preprocessor.transformers_[1][1]   # OneHotEncoder directly
cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])

# Combine all features
all_feature_names = np.concatenate([num_features, cat_features])

# Extract trained classifier
rf_model = pipeline.named_steps['clf']
importances = rf_model.feature_importances_

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance_df['Feature'][:15][::-1],
    feature_importance_df['Importance'][:15][::-1],
    color='skyblue'
)
plt.xlabel('Importance Score')
plt.title('Top 15 Feature Importances from Random Forest')
plt.tight_layout()
plt.show()


# In[39]:


import joblib
# from imblearn.pipeline import Pipeline as ImbPipeline # This import is not needed if you're not using ImbPipeline for the main model object

# Assuming 'model' is your trained RandomForestClassifier object
# (This object is created and trained in your provided training code snippet)

model_filename = 'random_forest_model.joblib' # Changed filename to reflect direct model saving

try:
    joblib.dump(model, model_filename) # Changed 'model_pipeline' to 'model'
    print(f"Model successfully saved to {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")


# In[41]:


get_ipython().run_line_magic('whos', 'DataFrame')


# In[42]:


import os

# Ensure 'data' folder exists
os.makedirs("data", exist_ok=True)

# Save the cleaned dataset
X.to_csv("data/cleaned_customer_booking.csv", index=False)


# #### Feature Importance Summary
# 
# Top features influencing customer purchase behavior:
# 
# - **route_SIN-DPS** – Most influential; route-specific patterns matter.
# - **flight_hour** – Timing impacts buying decisions.
# - **booking_origin_SG** – Region of booking is key.
# - **purchase_lead** – Advance booking period is important.
# - **trip_type_R** – Trip type (return/one-way) affects behavior.
# 
# These insights highlight the importance of **geographic** and **temporal** factors in predicting customer actions.

# # British Airways Task 2: Predicting Customer Buying Behaviour
# 
# ---
# 
# ## Objective
# 
# To develop a machine learning model that accurately predicts which customers are likely to make a purchase, enabling British Airways to optimize its marketing and engagement strategies.
# 
# ---
# 
# ## Data Overview
# 
# - **Total records:** ~10,000  
# - **Target variable:** `Purchase` (1 = made a purchase, 0 = did not purchase)  
# - **Class imbalance observed:**
#   - **Non-buyers (0):** ~85%
#   - **Buyers (1):** ~15%
# 
# ---
# 
# ## Data Preparation & Processing
# 
# - **Encoding:** Categorical variables encoded using `OneHotEncoder` within a `ColumnTransformer`.
# - **Scaling:** Numerical variables standardized with `StandardScaler`.
# - **Pipeline:** Combined preprocessing with Random Forest model in a `Pipeline`.
# - **Train-test split:** 80/20 split used for model validation.
# 
# ---
# 
# ## Addressing Class Imbalance
# 
# - **SMOTE (Synthetic Minority Oversampling Technique)** applied **only to training data**.
# - Created synthetic samples for the minority class (buyers), balancing the training set.
# 
# ---
# 
# ## Model Used: Random Forest Classifier
# 
# - **Reason for choice:** Non-linear model, robust to overfitting, handles mixed feature types well.
# 
# ---
# 
# ## Evaluation Metrics on Test Set
# 
# | Metric     | Score   |
# |------------|---------|
# | Accuracy   | 0.8250  |
# | Precision  | 0.3736  |
# | Recall     | 0.2470  |
# | F1 Score   | 0.2974  |
# 
# ---
# 
# ## Classification Report
# 
# | Class           | Precision | Recall | F1-score | Support |
# |-----------------|-----------|--------|----------|---------|
# | 0 (Non-buyers)  | 0.87      | 0.93   | 0.90     | 8379    |
# | 1 (Buyers)      | 0.37      | 0.25   | 0.30     | 1478    |
# 
# ---
# 
# ## Confusion Matrix
# 
# |                 | Predicted: 0 | Predicted: 1 |
# |-----------------|--------------|--------------|
# | **Actual: 0**   | 7767 (TN)    | 612 (FP)     |
# | **Actual: 1**   | 1113 (FN)    | 365 (TP)     |
# 
# ---
# 
# ## Insights
# 
# - Model performs **very well on non-buyers**, correctly identifying ~93% of them.
# - For buyers, **precision is acceptable (37%)**, but **recall is low (25%)**, indicating many buyers are being missed.
# - F1 Score is modest (0.30), reflecting a trade-off between precision and recall.
# - This suggests the model is **more conservative**, avoiding false positives but missing some actual buyers.
# 
# ---
# 
# ## Recommendations
# 
# 1. **Improve Recall:**
#    - Try other algorithms (e.g., XGBoost, Logistic Regression).
#    - Experiment with different resampling ratios in SMOTE.
#    - Use cost-sensitive learning (class weights).
# 
# 2. **Threshold Tuning:**
#    - Adjust prediction threshold to optimize for recall or F1 based on business priorities.
# 
# 3. **Feature Engineering:**
#    - Create domain-specific features (e.g., past travel frequency, time since last booking).
# 
# 4. **A/B Testing in Marketing:**
#    - Use current model for a test campaign and measure conversion lift.
# 

# In[ ]:




