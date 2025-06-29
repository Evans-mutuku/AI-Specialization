# Import necessary libraries
from flask import Flask, request, jsonify
import pickle

# Save the model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Create a Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    cleaned_review = preprocess_text(data)
    X = vectorizer.transform([cleaned_review])
    prediction = model.predict(X)
    return jsonify({'sentiment': prediction[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)