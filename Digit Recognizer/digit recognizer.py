import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    
from keras.utils import to_categorical

""" converting from panda df to numpy
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data_np = train_data.to_numpy()
test_data_np = test_data.to_numpy()

print(train_data_np.shape)
print(test_data_np.shape)


train_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
test_data = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

#print(train_data.shape)
#print(test_data.shape)
#print(train_data[0:4,0])

y_train = train_data[:,0]
X_train = train_data[:,1:]
"""
#Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate the features and label
y_train=train_data['label']
X_train=train_data.drop('label',axis=1)

#Preprocess the data
X_train = X_train.values.reshape(-1,28,28,1)/255.0
y_train = to_categorical(y_train)

"""
# Extract the pixel values
pixels = train_data.drop('label', axis=1).values.flatten()

# Plot the histogram
plt.hist(pixels, bins=16, range=(0, 256))
plt.title('Histogram of pixel values')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()
"""
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2)



"""
print("Training data shape: ", X_train.shape)
print("Training labels shape: ", y_train.shape)
print("Validation data shape: ", X_val.shape)
print("Validation labels shape: ", y_val.shape)

plt.hist(train_data['label'], bins=10, range=(-0.5, 9.5), align='mid')
plt.xticks(range(10))
plt.xlabel('Digit')
plt.ylabel('Frequency')
plt.title('Frequency of Digits in Training Data')
plt.show()
X_train.shape[1]
""" 


# Define the model
model= tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    
])
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='SparseCategoricalCrossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(X_train,y_train, epoch=100,batch_size=50,validation_data=(X_val,y_val))

# predict model

X_test = test_data.values.reshape(-1,28,28,1)/255.0
y_pred=model.predict(X_test)

#save the predictions to a CSV file
submission = pd.DataFrame({'ImageID':test_data.index+1,'Label':y_pred.argmax(axis=1)})
submission.to_csv('submission.csv',index=False)