import joblib
from sklearn.linear_model import LogisticRegression

# Assuming you have already trained your model and it is stored in the 'model' variable

# Create an instance of LogisticRegression
model = LogisticRegression()

# Save the trained model to a file using joblib
joblib.dump(model, 'sql New.joblib')

