from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

# Load the wine quality dataset (assuming it's in a CSV file)

#data = pd.read_csv('/home/mussie/Music/home projects/CI CD in ml/winequalityN.csv')
data = pd.read_csv('winequalityN.csv')

#fill missing values
data.fillna(0,inplace=True)

# Convert the 'type' column into dummy variables
data = pd.get_dummies(data, columns=['type'])


# Split the dataset into features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the random forest regressor
model = RandomForestRegressor(n_estimators=150, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Report training set score
train_score = model.score(X_train, y_train) * 100
# Report test set score
test_score = model.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print('saved successfully')
