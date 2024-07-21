from test import PredictClass
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
predictor = PredictClass()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.json
        text = data['text']
        predicted_class = predictor.input_en(text)
        response = {
            'predicted_class': predicted_class
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
