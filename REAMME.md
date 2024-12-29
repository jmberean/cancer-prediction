# Breast Cancer Classification with Neural Network

This project demonstrates how to preprocess a dataset and build a deep learning model using TensorFlow/Keras to classify breast cancer as malignant or benign.

## Dataset
The dataset used in this project is assumed to be in a CSV file named `Cancer_Data.csv`. It contains various features describing tumor characteristics and a diagnosis column indicating whether the tumor is malignant (`M`) or benign (`B`).

## Workflow

1. **Data Loading**  
   The dataset is loaded into a pandas DataFrame.

2. **Data Cleaning**  
   - Replace infinite values with `NaN`.  
   - Drop columns containing only `NaN` values.  

3. **Feature Selection and Encoding**  
   - Drop unnecessary columns such as `id`.  
   - Map the `diagnosis` column to binary values:  
     - `M` -> `1` (Malignant)  
     - `B` -> `0` (Benign)

4. **Handling Missing Values**  
   Replace missing values in each feature column with the mean value.

5. **Standardization**  
   Standardize feature values to have a mean of 0 and a standard deviation of 1.

6. **Train-Test Split**  
   Split the data into training (80%) and testing (20%) sets.

7. **Model Architecture**  
   A neural network with the following layers:
   - Dense layer with 128 neurons and ReLU activation
   - Dense layer with 64 neurons and ReLU activation
   - Dense layer with 1 neuron and sigmoid activation for binary classification

8. **Compilation and Training**  
   - Optimizer: Adam with a learning rate of 0.001 and gradient clipping  
   - Loss function: Binary Crossentropy  
   - Metric: Accuracy  
   - Train the model for 100 epochs.

9. **Evaluation**  
   Evaluate the model's performance on the test set using loss and accuracy metrics.

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- TensorFlow
- scikit-learn

Install the required libraries using pip:

pip install pandas numpy tensorflow scikit-learn

## Running the Code

Place the Cancer_Data.csv file in the same directory as the script.

## Run the script:

python breast_cancer_classification.py

## Output

The script prints the cleaned dataset and displays training progress.

After training, the model's performance is evaluated and printed.

## Customization

Adjust the test_size parameter in train_test_split for different train-test ratios.

Modify the model architecture or hyperparameters as needed.