import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import xgBOOST as xgb
from xgboost import XGBClassifier


# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.info()
test_data.info()
# Preprocess the data
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing values by imputation
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables to numeric
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    # Create new features or feature engineering
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = np.where(data['FamilySize'] == 1, 1, 0)
    data['AgeClass'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, np.inf], labels=[1, 2, 3, 4])
    data['FareClass'] = pd.cut(data['Fare'], bins=[0, 10, 25, 50, np.inf], labels=[1, 2, 3, 4])
    data['AgeClass'] = data['AgeClass'].astype(float)
    data['FareClass'] = data['FareClass'].astype(float)

    # Fill missing values by imputation for 'FareClass'
    data['FareClass'].fillna(data['FareClass'].mode()[0], inplace=True)

    # Drop original columns
    data = data.drop(['Age', 'Fare', 'SibSp', 'Parch'], axis=1)
    
    return data

# Preprocess the training data
train_data = preprocess_data(train_data)

# Split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['Survived'])

# Separate the features and labels
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']
X_val = val_data.drop(['Survived'], axis=1)
y_val = val_data['Survived']

# Preprocess the test data
X_test = preprocess_data(test_data)


# Define the model
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=300, max_depth=5, random_state=42)

# Train the model
model.fit(X_train, y_train)


# Calculate the confusion matrix and classification report for validation set
y_pred_val = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred_val)
cr = classification_report(y_val, y_pred_val)
print('Confusion Matrix for the validation set:\n', cm)
print('Classification Report for the validation set:\n', cr)

# Predict on the test data
y_pred_test = model.predict(X_test)
y_pred_test = np.round(y_pred_test).astype(int)


# Calculate the accuracy of the model on the training and validation sets
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_pred_val)

# Plot the accuracy on a graph
plt.plot([train_acc, val_acc], marker='o')
plt.xticks([0, 1], ['Training', 'Validation'])
plt.ylabel('Accuracy')
plt.show()

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Save the predictions to a CSV file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred_test})
submission.to_csv('submission.csv', index=False)
