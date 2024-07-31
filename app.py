from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models
model_red = joblib.load('model_red.pkl')
model_white = joblib.load('model_white.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    wine_type = data['wine_type']
    features = np.array(data['features']).reshape(1, -1)

    if wine_type == 'red':
        prediction = model_red.predict(features)
    elif wine_type == 'white':
        prediction = model_white.predict(features)
    else:
        return jsonify({'error': 'Invalid wine type. Use "red" or "white".'})

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
