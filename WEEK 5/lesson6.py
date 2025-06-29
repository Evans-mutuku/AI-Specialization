from flask import Flask, request, jsonify
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['review']
    cleaned_review = preprocess_text(data)
    X = vectorizer.transform([cleaned_review])
    prediction = model.predict(X)
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)