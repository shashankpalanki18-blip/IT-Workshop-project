from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained AI model
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    # This serves your web interface
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the hand coordinates sent from the browser
        data = request.get_json()
        landmarks = data.get('landmarks')
        
        # Format the data and make a prediction
        features = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Send the translated letter back to the browser
        return jsonify({'letter': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)