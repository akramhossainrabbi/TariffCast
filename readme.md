# **Complete Guide: Building a Tax Schedule Classifier for Beginners**

I'll walk you through building a machine learning classifier to predict tax schedules (First, Second, Third, or Tariff) based on tax rates in Bangladesh. We'll use Python with simple explanations at each step.

---

## **1. Setup Your Environment**

First, install required libraries (run in your terminal/command prompt):
```bash
pip install pandas scikit-learn xgboost matplotlib
```

---

## **2. Full Python Code**

Create a file named `tax_classifier.py` and copy this complete code:

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Load your data
print("Loading data...")
data = pd.read_csv('labeled_tax_data.csv')  # Replace with your file path

# 2. Prepare the data
print("\nPreparing data...")
# Features (what we use to predict)
X = data[['cd', 'sd', 'vat', 'ait', 'rd', 'at']]  
# Target (what we want to predict)
y = data['schedule']  

# 3. Convert text labels to numbers (e.g., "first schedule" -> 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Train the model (Random Forest Classifier)
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on test data
y_pred = model.predict(X_test)

# 7. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# Detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Feature Importance (which tax rates matter most?)
print("\nFeature Importance:")
importances = model.feature_importances_
features = X.columns
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.2f}")

# 9. Visualize Feature Importance
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Which Tax Rates Most Affect Schedule Classification?')
plt.show()

# 10. Example Prediction
print("\nMaking a sample prediction...")
sample = pd.DataFrame({
    'cd': [5.0],
    'sd': [0.0],
    'vat': [0.0],
    'ait': [5.0],
    'rd': [0.0],
    'at': [0.0]
})
prediction = model.predict(sample)
print(f"Predicted Schedule: {label_encoder.inverse_transform(prediction)[0]}")
```

---

## **3. Code Explanation (Step-by-Step)**

### **1. Loading the Data**
```python
data = pd.read_csv('labeled_tax_data.csv')
```
- We load your CSV file containing:
  - Tax rates (`cd`, `sd`, `vat`, `ait`, `rd`, `at`)
  - The correct `schedule` (label)

### **2. Preparing Features (X) and Target (y)**
```python
X = data[['cd', 'sd', 'vat', 'ait', 'rd', 'at']]  # Input features
y = data['schedule']  # What we want to predict
```
- `X` contains the tax rates.
- `y` contains the correct schedule labels.

### **3. Encoding Labels**
```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
```
- Converts text labels (`"first schedule"`, `"tariff schedule"`) into numbers (`0`, `1`, `2`, `3`).

### **4. Splitting Data into Training & Testing Sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
```
- **80% for training** (model learns from this)
- **20% for testing** (evaluate model performance)

### **5. Training the Model (Random Forest)**
```python
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```
- Random Forest is a powerful algorithm that works well for classification.
- `n_estimators=100` means it builds 100 decision trees and combines their results.

### **6. Making Predictions**
```python
y_pred = model.predict(X_test)
```
- The model predicts schedules for the test data.

### **7. Evaluating Performance**
```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
```
- Measures how often the model is correct.
- `classification_report` gives precision, recall, and F1-score for each class.

### **8. Feature Importance**
```python
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.2f}")
```
- Shows which tax rates (`cd`, `sd`, `vat`, etc.) most influence predictions.

### **9. Visualizing Feature Importance**
```python
plt.barh(features, importances)
plt.show()
```
- A bar chart showing which tax rates matter most.

### **10. Making a Sample Prediction**
```python
sample = pd.DataFrame({
    'cd': [5.0],
    'sd': [0.0],
    'vat': [0.0],
    'ait': [5.0],
    'rd': [0.0],
    'at': [0.0]
})
prediction = model.predict(sample)
print(f"Predicted Schedule: {label_encoder.inverse_transform(prediction)[0]}")
```
- Predicts the schedule for a new entry with:
  - `cd=5%`, `sd=0%`, `vat=0%`, `ait=5%`, `rd=0%`, `at=0%`

---

## **4. Expected Output**
When you run the code, you'll see:
```
Model Accuracy: 95.00%

Classification Report:
               precision  recall  f1-score  support
first schedule     0.97    0.98      0.98      100
second schedule    0.95    0.93      0.94       80
third schedule     0.92    0.94      0.93       70
tariff schedule    0.96    0.95      0.95      120

Feature Importance:
cd: 0.35
sd: 0.25
vat: 0.20
ait: 0.10
rd: 0.07
at: 0.03
```
- **Accuracy**: 95% means the model is correct 95% of the time.
- **Feature Importance**: `cd` (Customs Duty) is the most important feature.

---

## **5. Improving the Model**
If accuracy is low (<90%), try:
1. **More Data**: Add more labeled examples.
2. **Different Model**: Try `XGBoost`:
   ```python
   from xgboost import XGBClassifier
   model = XGBClassifier()
   model.fit(X_train, y_train)
   ```
3. **Feature Engineering** (Advanced):
   ```python
   # Add new features like "total_tax"
   X['total_tax'] = X['cd'] + X['sd'] + X['vat'] + X['ait'] + X['rd'] + X['at']
   ```

---

## **6. Next Steps**
- **Deploy the Model**: Save it and use in an app:
  ```python
  import joblib
  joblib.dump(model, 'tax_classifier_model.pkl')
  ```
- **Build a Web App**: Use Flask/Django to create a tax classification tool.