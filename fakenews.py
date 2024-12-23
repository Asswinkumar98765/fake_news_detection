import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report # Load the dataset
def load_data(file_path):
#Load the dataset from a CSV file.
#The dataset must have 'text' and 'label' columns."""
    try:
        data = pd.read_csv(file_path) 
        return data
    except FileNotFoundError:
        print("Error: Dataset file not found.") 
        exit()
# Preprocess the data
def preprocess_data(data):

#Preprocess the data: clean text and vectorize using TF-IDF. """
# Remove missing values data = data.dropna()
# Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(data['text'])
    y = data['label'] 
    return X, y, tfidf

# Train the model
def train_model(X, y):

#Split data into training and testing sets and train the model. """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    model = LogisticRegression() 
    model.fit(X_train, y_train) 
    return model, X_test, y_test
# Evaluate the model
def evaluate_model(model, X_test, y_test):

#Evaluate the trained model using accuracy and classification report. """
    y_pred = model.predict(X_test) 
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred)) 
    print("\nClassification Report:\n", classification_report(y_test, y_pred)) # Predict new input d
def predict_news(text, model, tfidf):

#Predict if a news article is fake or real. """
    text_vectorized = tfidf.transform([text])  
    prediction = model.predict(text_vectorized) 
    return "Real" if prediction[0] == 1 else "Fake" # Main driver code
if __name__ == "__main__":
# Path to dataset (replace with your dataset file path)

    dataset_path = "path_to_dataset.csv" # Example: "fake_news.csv"

# Step 1: Load the dataset
data = load_data(dataset_path) # Step 2: Preprocess the data
X, y, tfidf = preprocess_data(data) # Step 3: Train the model
model, X_test, y_test = train_model(X, y) # Step 4: Evaluate the model evaluate_model(model, X_test, y_test)

# Step 5: Predict on new input
example_text = "Breaking news: AI can now predict the weather more accurately than humans!"
result = predict_news(example_text, model, tfidf)
print(f"\nThe statement: '{example_text}' is classified as: {result}")