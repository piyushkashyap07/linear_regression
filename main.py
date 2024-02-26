import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assuming your dataset is in a file named 'train.csv'
# Adjust the file path accordingly
file_path=r'C:\Piyush\Scripts\Machine-Learning\Linear_regression\0000000000002329_training_diabetes_x_y_train.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Extract features (columns 0 to 9) and target (column 10 and 11)
X = df.iloc[:, :11]
y = df.Y

# Assuming you want to predict the first target column (column 10)
# target_column_to_predict = 200
# y_target = df.iloc[:, target_column_to_predict]

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# You can use the trained model to make predictions on new data
# For example, assuming you have a new data frame named 'new_data'
# new_data_predictions = model.predict(new_data)
