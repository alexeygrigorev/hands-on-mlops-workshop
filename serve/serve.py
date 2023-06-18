import os
import mlflow
from flask import Flask, request, jsonify


MODEL_VERSION = os.getenv('MODEL_VERSION')
MODEL_URI = os.getenv('MODEL_URI')

model = mlflow.pyfunc.load_model(MODEL_URI)


def prepare_features(ride):
    features = {}
    features['PULocationID'] = ride['PULocationID']
    features['DOLocationID'] = ride['DOLocationID']
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    body = request.get_json()
    ride = body['ride']
    ride_id = body['ride_id']

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'prediction': {
            'duration': pred,
        },
        'ride_id': ride_id,
        'version': MODEL_VERSION,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)