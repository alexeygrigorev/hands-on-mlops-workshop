# Hands-on MLOps Guide

MLOps Hands-on Guide: From Training to Deployment and Monitoring

Plan:

* Part 0: Preparing the environment
* Part 1: Introduction to MLOps
* Part 2: Experiment tracking & model registry
* Part 3: Training pipelines
* Part 4: Deployment
* Part 5: Monitoring


This is stripped-down version of
[MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 


About the instructor:

* Founder of [DataTalks.Club](https://datatalks.club/) - community of 22k data enthusiasts
* Author of [ML Bookcamp](https://mlbookcamp.com/)
* Instructor of [ML Zoomcamp](http://mlzoomcamp.com/) and [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 
* Connect with me on [LinkedIn](https://www.linkedin.com/in/agrigorev/) and [Twitter](https://twitter.com/Al_Grigor) 


## Part 0: Preparing the environment

> Instead of creating a virtual machine, you can use your own Ubuntu 
> machine. 
> The content of this workshop will also work on Windows and MacOS,
> but it's best to use Ubuntu

### Creating a virtual machine on AWS (optional)

* Name: "hands-on-mlops"
* Ubuntu 22.04 LTS, AMD64 architecture 
* Recommended: t2.large (2 CPU and 8 GiB RAM)
* Create or select an existing key (note the name) 
* Create a security group
    * Allow SSH traffic from everywhere
* 20 GB storage
* Give admin rights to the instance profile of the VM

> Note: allowing traffic from everywhere and giving admin rights 
> to the instance is dangerous in production settings. Don't do 
> this at work and consult your security department for the 
> best setup. 


Configure SSH: open `~/.ssh/config` and add this

```
Host hands-on-mlops
    HostName ec2-3-253-31-99.eu-west-1.compute.amazonaws.com
    User ubuntu
    IdentityFile c:/Users/alexe/.ssh/razer.pem
    StrictHostKeyChecking no
```

Log in:

```bash
ssh hands-on-mlops
```

If you use VS Code, you can connect to this instance with the 
["Remote - SSH" extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)

Later, we will use port-forwarding to open things locally. If you 
don't use VS Code, you can do port-forwarding with ssh:

```
Host hands-on-mlops
    HostName ec2-3-253-31-99.eu-west-1.compute.amazonaws.com
    User ubuntu
    IdentityFile c:/Users/alexe/.ssh/razer.pem
    StrictHostKeyChecking no
    LocalForward 8888 localhost:8888
```

### Installing all the necessary software and libraries 

* Anaconda / [miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Docker


```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh
```

Log out/in and test if installation is successful:

```bash
which python
python -V
```

We should have Python 3.10.10 

Next, install Docker:


```bash
sudo apt update
sudo apt install docker.io

# to run docker without sudo
sudo groupadd docker
sudo usermod -aG docker $USER
```

Log out and in and then test that docker runs properly:

```bash
docker run hello-world
```

### Clone the workshop repo

```bash
git clone https://github.com/alexeygrigorev/hands-on-mlops-workshop.git
```


## Part 1: Introduction to MLOps

### What is MLOps

Poll: What's MLOps?

* https://datatalks.club/blog/mlops-10-minutes.html


### Preparation

* We'll start with the model [we already trained](train/duration-prediction-starter.ipynb)
* Copy this notebook to "duration-prediction.ipynb"
* This model is used for preducting the duration of a taxi trip

We'll start with preparing the environement for the workshop

```bash
pip install pipenv 
```


Run poll: "Which virtual environment managers have you used"

Options:

- Conda
- Python venv
- Pipenv
- Poetry
- Other
- Didn't use any

Create a directory (e.g. "train") and run there

```bash
pipenv --python=3.10
```

> **Note** if you have conda and don't have Python 3.10.10,
> you can install it using this command:
> `conda create --name py3-10-10 python=3.10.10`
> and then you can specify the path to your python
> executable: `pipenv install --python=pipenv install --python=/c/Users/alexe/anaconda3/envs/py3-10-10/python.exe` (or `~/anaconda3/envs/py3-10-10/bin/python`)


Install the dependencies

```bash
pipenv install scikit-learn==1.2.2 pandas pyarrow seaborn
pipenv install --dev jupyter
```

On Linux you might also need to instal `pexpect` for jupyter:

```bash
pipenv install --dev jupyter pexpect
```


We will use the data from the [NYC TLC website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

* Train: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
* Validation: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet

Run the notebook

```bash
pipenv run jupyter notebook
```

Forward the 8888 port if you're running the code remotely

Now open http://localhost:8888/

## Part 2: Experiment tracking & model registry

### Experiment tracking

First, let's add mlflow for tracking experiments. 

```bash
pipenv install mlflow==2.4.1 boto3
```

Run MLFlow locally (replace it with your bucket name)

```bash
pipenv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://mlflow-models-alexey
```

(add 5000 to port forwarding if you're doing it remotely)

Open it at http://localhost:5000/


Connect to the server from the notebook

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")
```

Log the experiment:

```python
with mlflow.start_run():
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    mlflow.log_params({
        'categorical': categorical,
        'numerical': numerical,
    })

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(rmse)
    mlflow.log_metric('rmse', rmse)

    with open('dict_vectorizer.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact('dict_vectorizer.bin')
    
    mlflow.sklearn.log_model(lr, 'model')

    print(f'run ID: {mlflow.active_run().info.run_id}')
```

Go to MLFlow UI and find the latest run: 

```
attributes.run_id='f2c07e306f9a44308bafda35e22fc9f1'
```

Or find it using a direct link: http://localhost:5000/#/experiments/1/runs/f2c07e306f9a44308bafda35e22fc9f1


Now let's add a parameter:

```python
model_params = {
    'fit_intercept': True
}
mlflow.log_params(model_params)

lr = LinearRegression(**model_params)
```

Replace it with a pipeline:


```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LinearRegression(**model_params)
)

pipeline.fit(train_dicts, y_train)
y_pred = pipeline.predict(val_dicts)

mlflow.sklearn.log_model(pipeline, 'model')
```

### Loading the model

```python
logged_model = 'runs:/7c373fc9626549ed91cebb714b07e60a/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

records = df[categorical + numerical].to_dict(orient='records')

loaded_model.predict(records)
```

### Model registry

Register the model as "trip_duration" model, stage "staging"

Let's get this model

```python
model = mlflow.pyfunc.load_model('models:/trip_duration/staging')
```

And use it:

```python
y_pred = model.predict(val_dicts)
```

In some cases we don't want to depend on the MLFlow model registry
to be always available. In this case, we can get the S3 path
of the model and use it directly for initializing the model

```bash
MODEL_METADATA=$(pipenv run python storage_uri.py \
    --tracking-uri http://localhost:5000 \
    --model-name trip_duration \
    --stage-name staging)
echo ${MODEL_METADATA}
```

Now we can use the storage URL to load the model:

```python
model = mlflow.pyfunc.load_model(storage_url)
y_pred = model.predict(val_dicts)
```

## Part 3: Training pipelines

### Creating a training script

Convert the notebook to a script 

```bash
pipenv run jupyter nbconvert --to=script duration-prediction.ipynb
```

Rename the file to `train.py` and clean it

Run it:

```bash 
pipenv run python train.py
```

## Part 4: Deployment

Poll: "What can we use for serving an ML model?"

### Web service (Flask)

Now let's go to the `serve` folder and create a virtual 
environment

```bash
pipenv --python=3.10
pipenv install scikit-learn==1.2.2 mlflow==2.4.1 boto3 flask gunicorn
```

Create a Flask app

```python
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

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'prediction': {
            'duration': pred,
        }
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
```

Run it:

```bash
MODEL_METADATA=$(pipenv run python storage_uri.py \
    --tracking-uri http://localhost:5000 \
    --model-name trip_duration \
    --stage-name staging)

echo ${MODEL_METADATA} | jq

export MODEL_VERSION=$(echo ${MODEL_METADATA} | jq -r ".run_id")
export MODEL_URI=$(echo ${MODEL_METADATA} | jq -r ".source")

pipenv run python serve.py
```

Test it:

```bash
REQUEST='{
    "ride": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    }
}'
URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL}
```

### Dockerize 


### Deployment

Now package the model with Docker and deploy it
(outside of the scope for this tutorial).


## Part 5: Monitoring


## Links

### Also see

* https://github.com/DataTalksClub/mlops-zoomcamp
* https://github.com/alexeygrigorev/lightweight-mlops-zoomcamp
* https://github.com/alexeygrigorev/ml-observability-workshop