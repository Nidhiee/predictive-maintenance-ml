

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("/content/predictive_maintenance.csv")

# Select features and target
X = data[['Air temperature [K]', 'Process temperature [K]',
          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

