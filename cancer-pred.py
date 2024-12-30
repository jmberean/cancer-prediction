import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.impute import SimpleImputer

# Load dataset
dataset = pd.read_csv('Cancer_Data.csv')
print(dataset)

# Replace infinite values and drop columns with all NaNs
dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
print(dataset)

# Separate features (X) and labels (y), map diagnosis to binary values
x = dataset.drop(columns=["id", "diagnosis"])
y = dataset["diagnosis"].map({'M': 1, 'B': 0})

# Handle missing values and standardize features
x = SimpleImputer(strategy='mean').fit_transform(x)
x = StandardScaler().fit_transform(x)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a sequential neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs=100)

# Evaluate the model
model.evaluate(x_test, y_test)
