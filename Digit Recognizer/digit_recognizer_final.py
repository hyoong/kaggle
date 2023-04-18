import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    
from keras.utils import to_categorical

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y_train = train_data['label']
X_train = train_data.drop('label', axis=1)

# Preprocess the data
X_train = X_train.values.reshape(-1,28,28,1)/255.0
y_train = to_categorical(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_val, y_val))

# Predict model
X_test = test_data.values.reshape(-1,28,28,1)/255.0
y_pred = model.predict(X_test)

# Save the predictions to a CSV file
submission = pd.DataFrame({'ImageId':test_data.index+1,'Label':y_pred.argmax(axis=1)})
submission.to_csv('submission.csv', index=False)
