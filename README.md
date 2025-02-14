import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import LabelEncoder, StandardScaler
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import classification_report, confusion_matrix
 # Load the dataset
 data = pd.read_csv('/content/user_behavior_dataset.csv')
 # Display the first few rows of the dataset
 print(data.head())
 # Check for missing values
 print(data.isnull().sum())
 # Basic statistics
 print(data.describe())
 # Visualize the distribution of numerical features
sns.pairplot(data)
 plt.show()
 print(data.columns)
 import seaborn as sns
 import matplotlib.pyplot as plt
 # Assuming 'data' is your DataFrame
 # Now try to create the countplot with the correct column name
 sns.countplot(x='User Behavior Class', data=data) # Use the correct
 column name
 plt.title('Count of User Behavior Class') # Optional: Add a title
 plt.xlabel('User Behavior Class') # Optional: Label for x-axis
 plt.ylabel('Count') # Optional: Label for y-axis
 plt.xticks(rotation=45) # Optional: Rotate x-axis labels if needed
 plt.show()
 # Handle missing values (if any)
 data.fillna(method='ffill', inplace=True)
 # Encode categorical variables
 label_encoder = LabelEncoder()
 data['Device Model'] = label_encoder.fit_transform(data['Device Model'])
 data['Operating System'] = label_encoder.fit_transform(data['Operating
 System'])
 data['Gender'] = label_encoder.fit_transform(data['Gender'])
 # Features and target variable
X = data.drop(['User ID', 'User Behavior Class'], axis=1)
 y = data['User Behavior Class']
 # Standardize the features
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 # Split the dataset into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
 test_size=0.2, random_state=42)
 # Initialize and train the model
 # Make predictions
 y_pred = model.predict(X_test)
 # Evaluate the model
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 from sklearn.metrics import accuracy_score
 # Calculate accuracy
 accuracy = accuracy_score(y_test, y_pred)
 print(f'Accuracy: {accuracy * 100:.2f}%')
 # Visualizing the confusion matrix
 plt.figure(figsize=(8, 6))
 sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
 cmap='Blues')
 plt.title('Confusion Matrix')
plt.xlabel('Predicted')
 plt.ylabel('True')
 plt.show()
 from sklearn.model_selection import train_test_split
 # Example of splitting the data
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
 random_state=42)
 print(y.value_counts())
 from sklearn.metrics import classification_report, confusion_matrix
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 from sklearn.model_selection import cross_val_score
 scores = cross_val_score(model, X, y, cv=5) # Replace 'model' with your
 model
 print(f'Cross-Validation Accuracy: {scores.mean() * 100:.2f}%')
 import numpy as np
 # Example: Add Gaussian noise to a feature
 noise = np.random.normal(0, 0.1, X_train.shape)
 X_train_noisy = X_train + noise
 from sklearn.tree import DecisionTreeClassifier
 model = DecisionTreeClassifier(max_depth=3) # Limit the depth to
 prevent overfitting
X_train_small = X_train.sample(frac=0.5, random_state=42) # Use only
 50% of the training data
 y_train_small = y_train[X_train_small.index]
 from sklearn.linear_model import Ridge
 model = Ridge(alpha=10) # Increase alpha for stronger regularization
 from sklearn.model_selection import cross_val_score
 scores = cross_val_score(model, X, y, cv=10) # 10-fold cross-validation
 print(f'Cross-Validation Accuracy: {scores.mean() * 100:.2f}%')
 # Visualization of Results
 # Visualizing the confusion matrix
 plt.figure(figsize=(8, 6))
 sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
 cmap='Blues')
 plt.title('Confusion Matrix')
 plt.xlabel('Predicted')
 plt.ylabel('True')
 plt.show()
r
