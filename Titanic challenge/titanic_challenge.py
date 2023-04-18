import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

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
    
    # Fill missing values by imputation for 'FareClass'
    data['FareClass'].fillna(data['FareClass'].mode()[0], inplace=True)

    # Drop original columns
    data = data.drop(['Age', 'Fare', 'SibSp', 'Parch'], axis=1)
    
    return data

# Preprocess the training data
train_data = preprocess_data(train_data)

# Split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2)

# Separate the features and labels
X_train = train_data.drop(['Survived'], axis=1)
y_train = train_data['Survived']
X_val = val_data.drop(['Survived'], axis=1)
y_val = val_data['Survived']

# Preprocess the test data
X_test = preprocess_data(test_data)

# Create a decorator to implement the MISH activation function.
@tf.function
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))



# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    
    tf.keras.layers.Dense(128,activation=mish),
    tf.keras.layers.Dense(64,activation=mish),
   
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
optimizer=tf.keras.optimizers.Adam(learning_rate=0.009)
model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

# Check for missing values in train_data
print(train_data.isna().sum())

# Check for missing values in val_data
print(val_data.isna().sum())

# Check for missing values in X_test
print(X_test.isna().sum())

# best result epochc=70 batch_szie=50
# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=50, validation_data=(X_val, y_val))



# Predict on the test data
y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Save the predictions to a CSV file
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred[:, 0]})
submission.to_csv('submission.csv', index=False)

