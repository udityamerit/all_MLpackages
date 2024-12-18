# Breast Cancer Prediction Using Multiple Machine Learning Models

## Overview
This project implements multiple machine learning algorithms to predict breast cancer diagnoses based on medical diagnostic data. The project compares the performance of various models, providing insights into which algorithms are most effective for this task. The complete code is available in the [model.ipynb](https://github.com/udityamerit/Breast-Cancer-Prediction-using-different-ML-models/blob/main/model.ipynb) file.

---

## Features
- **Dataset**: Features medical data such as `radius_mean`, `texture_mean`, `perimeter_mean`, and others. The target column is `diagnosis` (malignant or benign).
- **Algorithms**: Includes Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, and more.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and ROC-AUC.
- **Visualization**: Graphs and tables illustrate model comparisons.
- **Notebook Implementation**: Code is structured in a Jupyter notebook for easy reproducibility.

---

## Prerequisites
- Python (>= 3.8)
- Required libraries:
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn
  ```

---

## Workflow
1. **Data Preprocessing**:
   - Load and clean the dataset
   - Encode categorical variables
   - Normalize features using `StandardScaler`
   - Split data into training and testing sets

2. **Model Training and Evaluation**:
   - Implement multiple ML algorithms
   - Evaluate models using metrics like accuracy and ROC-AUC

3. **Visualization**:
   - Generate comparison graphs for model performance

---

## Dataset Description
The dataset includes:
- **Features**:
  - Mean values: `radius_mean`, `texture_mean`, `perimeter_mean`
  - Standard error: `radius_se`, `texture_se`, `perimeter_se`
  - Worst values: `radius_worst`, `texture_worst`, `perimeter_worst`
- **Target Variable**:
  - `diagnosis`: Malignant (`M`) or Benign (`B`)

---

## Code Snippets
### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
```

### 2. Data Preprocessing
```python
# Load dataset
data = pd.read_csv('breast_cancer_data.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Feature-target split
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 3. Training Models
#### Logistic Regression Example
```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluation
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

#### Random Forest Example
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation
y_pred_rf = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
```

---

## Results
### Model Performance
| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 96%      | 94%       | 95%    | 94.5%    | 97%     |
| Random Forest       | 98%      | 97%       | 96%    | 96.5%    | 99%     |
| SVM                 | 95%      | 93%       | 94%    | 93.5%    | 96%     |
| k-NN                | 92%      | 90%       | 91%    | 90.5%    | 93%     |

### Visualization
![Model Comparison](performance_graph.png)

---

## Conclusion
The Random Forest classifier achieved the highest accuracy and ROC-AUC, making it the most effective model for this dataset. Logistic Regression and SVM also performed well, indicating their suitability for medical diagnostic tasks.

---

## Future Enhancements
- Incorporate deep learning models for enhanced prediction accuracy
- Perform feature selection to reduce dimensionality
- Use cross-validation for more robust performance evaluation

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/udityamerit/Breast-Cancer-Prediction-using-different-ML-models
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook model.ipynb
   ```

---

## Acknowledgments
- The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
- Project inspired by real-world medical applications of machine learning.
