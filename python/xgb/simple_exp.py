import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Sample dataset with numerical and text data
data = pd.DataFrame({
    "text": ["This is good", "I love it", "Terrible experience", "Not bad", None, "Absolutely fantastic"],
    "numerical": [1.5, 2.3, -1.0, None, 0.5, 3.2],
    "category": ["A", "B", "A", "C", "B", None],
    "category_2": ["X", "Y", "X", "Z", "Y", "X"],  # Additional categorical column
    "label": [1, 1, None, 0, 1, 1]
})

print(data.head(2))
print(data.describe())

print("missing data")
print(data["label"].isnull().sum())
print(data["text"].isnull().sum())
print(data["numerical"].isnull().sum())
print(data["label"].value_counts())

def handle_missing_features(data):
        # Fill missing text data
    data["text"] = data["text"].fillna("").str.lower()  # Replace None with empty string

    # Fill missing numerical data
    mean_value = data["numerical"].mean()  # Compute mean of numerical column
    data["numerical"] = data["numerical"].fillna(mean_value)  # Replace NaN with mean

    # Fill missing categorical data
    data["category"] = data["category"].fillna("UNKOWN").str.upper() # Replace None with "Unknown"

    print("\nPreprocessed Data:")
    print(data)
    return data

data_cleaned = handle_missing_features(data)
data_cleaned = data.dropna(subset=["label"])  # Drop rows with missing labels

# Separate features and labels
text_data = data_cleaned["text"]
numerical_data = data_cleaned["numerical"].values.reshape(-1, 1)
labels = data_cleaned["label"]

# Text preprocessing using TF-IDF
tfidf = TfidfVectorizer(max_features=10)
text_features = tfidf.fit_transform(text_data).toarray()
print("text_features")
print(tfidf)
print(text_features[:2])
print(text_features.shape)
# Scale numerical features
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(numerical_data)
scaled_numerical = scaled_numerical.reshape(-1, 1)
print("scaled_numerical")
print(scaler)
print(scaled_numerical.shape)
print(scaled_numerical[:2])

# One-Hot Encoding for categorical
encoder = OneHotEncoder(sparse_output=False)
categorical_features = encoder.fit_transform(data_cleaned["category"].values.reshape(-1, 1))
print("categorical_features")
print(encoder)
print(categorical_features[:2]) 
print(categorical_features.shape)

# Combine text and numerical features
combined_features = np.hstack([text_features, scaled_numerical, categorical_features])

print(combined_features[:2])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# save models
import joblib
complete_model = {
    "model": xgb_model,
    "tfidf": tfidf,
    "scaler": scaler,
    "encoder": encoder
}
joblib.dump(complete_model, "./model_saved/complete_model.pkl")

# load models
loaded_model = joblib.load("./model_saved/complete_model.pkl")
xgb_model = loaded_model["model"]
tfidf = loaded_model["tfidf"]
scaler = loaded_model["scaler"]
encoder = loaded_model["encoder"]


data_infer = pd.DataFrame({
    "text": [None, "This is good", "I love it", "Terrible experience", "Not bad", "Absolutely fantastic"],
    "numerical": [None, 1.5, 2.3, -1.0, 0.5, 3.2],
    "category": [None, "A", "B", "A", "C", "B"]
})

y_test = [1, 1, 1, 0, 1, 1]
data_infer_cleaned = handle_missing_features(data_infer)    

text_data_infer = data_infer_cleaned["text"]
text_data_infer = tfidf.transform(text_data_infer).toarray()

numerical_data_infer = data_infer_cleaned["numerical"].values.reshape(-1, 1)
numerical_data_infer = scaler.transform(numerical_data_infer)

categorical_features_infer = encoder.transform(data_infer_cleaned["category"].values.reshape(-1, 1))

combined_features_infer = np.hstack([text_data_infer, numerical_data_infer, categorical_features_infer])
print(combined_features_infer.shape)

y_pred_infer = xgb_model.predict(combined_features_infer)
print(y_pred_infer)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_infer)
print(f"Accuracy: {accuracy:.2f}")