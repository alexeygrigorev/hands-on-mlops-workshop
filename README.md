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

* Founder of [DataTalks.Club](https://datatalks.club/) - community of 36k+ data enthusiasts
* Author of [ML Bookcamp](https://mlbookcamp.com/)
* Instructor of [ML Zoomcamp](http://mlzoomcamp.com/), [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) and [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)
* Connect with me on [LinkedIn](https://www.linkedin.com/in/agrigorev/) and [Twitter](https://twitter.com/Al_Grigor) 


## Part 0: Preparing the environment

Instead of creating a virtual machine, you can use your own Ubuntu 
machine.

The content of this workshop will also work on Windows and MacOS,
but it's best to use Ubuntu.

But if you don't use an EC2 instance, you will need to create 
an AWS user and explicitly pass your keys inside the docker containers.


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

### Gunicorn 

Let's run it with gunicord (won't work on Windows. it you're on Windows,
just skip it and jump to the Docker part) 


### Dockerize 

Create a `Dockerfile`:

```docker
FROM python:3.10.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["serve.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "serve:app"]
```

Build the image:

```bash
docker build -t trip_duration:v1 .
```

Run it:

```bash
docker run -it \
    -p 9696:9696 \
    -e MODEL_VERSION="${MODEL_VERSION}" \
    -e MODEL_URI="${MODEL_URI}" \
    trip_duration:v1
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

### Add more information

Later we will need more information about the request, e.g. ride_id
and the version of the model used for serving this request. Let's
add this information now:

```python
ride_id = body['ride_id']

result = {
    'prediction': {
        'duration': pred,
    },
    'ride_id': ride_id,
    'version': MODEL_VERSION,
}
```

Rebuild the image:

```bash
docker build -t trip_duration:v2 .
```

Run it:

```bash
docker run -it \
    -p 9696:9696 \
    -e MODEL_VERSION="${MODEL_VERSION}" \
    -e MODEL_URI="${MODEL_URI}" \
    trip_duration:v2
```

Test it:

```bash
REQUEST='{
    "ride": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    },
    "ride_id": "xyz"
}'
URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL} | jq
```


### Deployment

Now we can deploy it anywhere. For example, Elastic Beanstalk

```bash
pipenv install --dev awsebcli
```

Init the project:

```bash
pipenv run eb init -p docker trip_duration
```

To run locally, open `.elasticbeanstalk/config.yaml` and replace 

```yaml
  default_platform: 'Docker running on 64bit Amazon Linux 2'
  default_region: eu-west-1
```

Run it in Docker:

```bash
pipenv run eb local run \
    --port 9696 \
    --envvars MODEL_VERSION="${MODEL_VERSION}",MODEL_URI="${MODEL_URI}"
```

Now let's deploy it.

We need to create a role with read-only S3 access to the bucket with
the models

* Go to AIM policies, create "mlflow-bucket-read-only"
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:Get*",
          "s3:List*"
        ],
        "Resource": [
          "arn:aws:s3:::mlflow-models-alexey/*",
          "arn:aws:s3:::mlflow-models-alexey"
        ]
      }
    ]
  }
  ```
* Go to IAM roles, click create "ec2-mlflow-bucket-read-only" and attach "mlflow-bucket-read-only" 
* Note the instance-profile ARN (something like "arn:aws:iam::XXXXXXXX:instance-profile/ec2-mlflow-bucket-read-only")

Run 

```bash
pipenv run eb create trip-duration-env \
    --envvars MODEL_VERSION="${MODEL_VERSION}",MODEL_URI="${MODEL_URI}" \
    -ip arn:aws:iam::XXXXXXXX:instance-profile/ec2-mlflow-bucket-read-only
```

Wait for ~5 minutes

Note: to update the environment variables later, use 

```bash
pipenv run eb setenv MODEL_VERSION="${MODEL_VERSION}" MODEL_URI="${MODEL_URI}"
```

Test it:

```bash
REQUEST='{
    "ride": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    },
    "ride_id": "xyz"
}'
URL="http://trip-duration-env.eba-im4te5md.eu-west-1.elasticbeanstalk.com/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL} | jq
```

Now we can terminate it.

Note: there are many ways to deploy this model. EB is easy to show, 
but in practice you will probably use something like ECS or Kubernetes.
Regardless of the platform, once you package your code in a Docker
container, you can run it anywhere.


## Part 5: Monitoring

Now we will add monitoring to our deployment.


### Logging the predictions

Log the predictions to a Kinesis stream and save them in a data lake (s3)

Now let's modify our `serve.py` to add logging. We will log the
prediction to a kinesis stream, but you can use any other way of 
logging.

Create a kinesis stream, e.g. `duration_prediction_serve_logger`. 

Add logging:

```python
import json
import boto3

PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'duration_prediction_serve_logger')
kinesis_client = boto3.client('kinesis')

# in the serve function

prediction_event = {
    'ride_id': ride_id,
    'ride': ride,
    'features': features,
    'prediction': result,
    'version': MODEL_VERSION,
}

print(f'logging {prediction_event} to {PREDICTIONS_STREAM_NAME}...')

kinesis_client.put_record(
    StreamName=PREDICTIONS_STREAM_NAME,
    Data=json.dumps(prediction_event) + "\n",
    PartitionKey=str(ride_id)
)
```

Note:

* It's important to add  `+ "\n"` at the end, else everything will be in one line
* For the IAM role for the intance profile, we will need to add "PutRecord" permissions to make it work (tip: Use ChatGPT for that)

You might need to specify the AWS region when doing it:

```bash
export AWS_DEFAULT_REGION="eu-west-1"
```

Send a request to test it:

```bash
REQUEST='{
    "ride_id": "XYZ10",
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

We can check the logs

```bash
KINESIS_STREAM_OUTPUT='duration_prediction_serve_logger'
SHARD='shardId-000000000000'

SHARD_ITERATOR=$(aws kinesis \
    get-shard-iterator \
        --shard-id ${SHARD} \
        --shard-iterator-type TRIM_HORIZON \
        --stream-name ${KINESIS_STREAM_OUTPUT} \
        --query 'ShardIterator' \
)

RESULT=$(aws kinesis get-records --shard-iterator $SHARD_ITERATOR)

echo ${RESULT} | jq -r '.Records[0].Data' | base64 --decode
```


### Getting data from S3

Set up a batch job for pulling the data from S3 and analyzing it 

* Create an s3 bucket "duration-prediction-serve-logs"
* Enable firehose
* No data transformation (explore yourself)
* No data converstion (explore yourself)
* Destination: "s3://duration-prediction-serve-logs"
* Look at the files in the bucket

Send a few requests to test it:

```bash
for i in {1..20}; do
    REQUEST='{
        "ride_id": "XYZ10-'${i}'",
        "ride": {
            "PULocationID": '${i}',
            "DOLocationID": 102,
            "trip_distance": '${i}'
        }
    }'

    echo ${REQUEST}

    curl -X POST \
       -d "${REQUEST}" \
       -H "Content-Type: application/json" \
       ${URL} | jq
done
```

They will appear in ~5 minutes (depending on the buffer size/wait time)


### Preparing the env 

Now we will prepare another environment - for monitoring (in the `monitor` folder)

```bash
pipenv --python=3.10
pipenv install scikit-learn==1.2.2 pandas pyarrow mlflow==2.4.1 evidently jupyter python-dateutil boto3
```

### Preparing the mock data

We can't wait for long, so we simulated the traffic and put the 
data in the monitor/data folder. To generate it, run 
the `prepare-files.ipynb` notebook.


## Data drift report

Let's use Evidently to generate a simple visual report

First, load the reference data (data we used for training)

```python
df_reference = pd.read_parquet('data/2022/01/2022-01-full.parquet')
```

Evidently is quite slow when analyzing large datasets, so we should
take a sample:


```python
df_reference = pd.read_parquet('data/2022/01/2022-01-full.parquet')
```

Next, we load the "production" data. First, we load the trips:

```python
year = 2023
month = 1
day = 2

trips_file = f'data/{year:04d}/{month:02d}/{year:04d}-{month:02d}-{day:02d}.parquet'
df_trips = pd.read_parquet(trips_file)
```

Second, load the logs:

```python
logs_file = f'data/{year:04d}/{month:02d}/{year:04d}-{month:02d}-{day:02d}-predictions.jsonl'

df_logs = pd.read_json(logs_file, lines=True)

df_predictions = pd.DataFrame()
df_predictions['ride_id'] = df_logs['ride_id']
df_predictions['prediction'] = df_logs['prediction'].apply(lambda p: p['prediction']['duration'])
```

And merge them: 

```python
df = df_trips.merge(df_predictions, on='ride_id')
```

Now let's see if there's any drift. Import evidently:

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
```

Build a simple drift report:

```python
report = Report(metrics=[
    DataDriftPreset(columns=['PULocationID', 'DOLocationID', 'trip_distance']), 
])

report.run(reference_data=df_reference_sample, current_data=df_trips)

report.show(mode='inline')
```

In this preset report, it uses Jensen-Shannon distance to measure
the descrepancies between reference and production. While it says
that drift is detected, we should be careful about it and 
check other months.

We can tune it:

```python
report = Report(metrics=[
    DataDriftPreset(
        columns=['PULocationID', 'DOLocationID', 'trip_distance'],
        cat_stattest='psi',
        cat_stattest_threshold=0.2
        num_stattest='ks',
        num_stattest_threshold=0.2,
    ), 
])

report.run(reference_data=df_reference_sample, current_data=df_trips)
report.show(mode='inline')
```

Save the report as HTML:

```python
report.save_html(f'reports/report-{year:04d}-{month:02d}-{day:02d}.html')
``` 


We can also extract information from this report and use it 
for e.g. sending an alert:

```python
report_metrics = report.as_dict()['metrics']
report_metrics = {d['metric']: d['result'] for d in report_metrics}
drift_report = report_metrics['DataDriftTable']

if drift_report['dataset_drift']:
    # send alert
    print('drift detected!')
```

We won't implement the logic for sending alerts here, but you can
find a lot of examples online. Or use ChatGPT to help you.

You can generate these reports in your automatic pipelines and then
send them e.g. over email.

Let's create this pipeline.


## Creating a pipeline with Prefect

Now we'll use Prefect to orchestrate the report generation.

We will take the code we created and put it into a Python script. 
See `pipeline_sample.py` for details. 

Run prefect server:

```bash
pipenv run prefect config set PREFECT_UI_API_URL=http://127.0.0.1:4200/api
pipenv run prefect server start
```

Run the pipeline:

```bash
pipenv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
pipenv run python pipeline_sample.py
```


## Data checks

* Automate data checks as part of the prediction pipeline. Design a custom test suite for data quality, data drift and prediction drift.

Now we want to add data quality checks. We will start with simple
integrity checks: data types, missing values and so on.

They are also done via reports:


```python
from evidently.metrics import DatasetSummaryMetric, DatasetMissingValuesMetric

data_integrity_report = Report(metrics=[
    DatasetSummaryMetric(),
    DatasetMissingValuesMetric()
])

data_integrity_report.run(reference_data=df_reference_sample, current_data=df_trips)
data_integrity_report.show(mode='inline')
```

In addition to reports, we can add tests. You can think of these 
tests as unit/integration tests for software. They pass or fail, 
and if they fail, we get an alert - something is wrong with the data,
so we need to look at it.

```python
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset

data_stability = TestSuite(tests=[
    DataStabilityTestPreset(),
])

data_stability.run(reference_data=df_reference_sample, current_data=df_trips)
data_stability.show(mode='inline')
```

Let's tune the test:

```python
data_stability = TestSuite(tests=[
    TestNumberOfRows(gte=1000, lte=20000),
    TestNumberOfColumns(),
    TestColumnsType(),
    TestAllColumnsShareOfMissingValues(),
    TestNumColumnsOutOfRangeValues(),
    TestCatColumnsOutOfListValues(
        columns=['PULocationID', 'DOLocationID', 'trip_distance']
    ),
    TestNumColumnsMeanInNSigmas(),
])

data_stability.run(reference_data=df_reference_sample, current_data=df_trips)
data_stability.show(mode='inline')
```

We can add this to our pipeline too:

```python
test_results = data_stability.as_dict()['tests']

failed_tests = []

for test in test_results:
    status = test['status']
    if status == 'FAIL':
        failed_tests.append(test)

if len(failed_tests) > 0:
    print('tests failed:')
    print(failed_tests)
```

Examples:

* https://github.com/evidentlyai/evidently/blob/main/examples/sample_notebooks/evidently_tests.ipynb

## Model quality checks

We also have labels that come with delay - every time the ride 
ended, we can compare the predictions with the actual duration
and make some conclusions. If our model performance drifts, 
we can notice it and react (e.g. by retraining the model)


First, we need to prepare the data a bit:

```python
df_reference_sample = df_reference_sample.rename(columns={'duration': 'target'})
df = df.rename(columns={'duration': 'target'})
```

Now let's run the report:

```python
regression_performance_report = Report(metrics=[
    RegressionPreset(columns=['PULocationID', 'DOLocationID', 'trip_distance']),
])

regression_performance_report.run(reference_data=df_reference_sample, current_data=df)
regression_performance_report.show(mode='inline')
```

Note: for classification you can use this report:

```python
from evidently.metric_preset import ClassificationPreset
```


## Alerting

* Setting up slack/email alerts 


## Summary

* Wrapping up: summarizing key takeaways and reviewing recommended practices


## Links

### Also see

* https://github.com/DataTalksClub/mlops-zoomcamp
* http://mlzoomcamp.com/
* https://github.com/alexeygrigorev/lightweight-mlops-zoomcamp
* https://github.com/alexeygrigorev/ml-observability-workshop