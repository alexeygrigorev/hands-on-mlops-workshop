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
    * Allow SSH traffic from everywhere (just for this workshop)
    * Allow HTTP traffic
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

First, let's add mlflow for tracking experiments 

```bash
pipenv install mlflow boto3
```

Run MLFlow locally (replace it with your bucket name)

```bash
pipenv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://mlflow-models-alexey
```

Open it at http://localhost:5000/


Connect to the server from the notebook

```python
import mflow

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
```

Replace it with a pipeline:


```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LinearRegression()
)

pipeline.fit(train_dicts, y_train)
y_pred = pipeline.predict(val_dicts)

mlflow.sklearn.log_model(pipeline, 'model')
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

## Part 4: Deployment


Poll: "What can we use for serving an ML model?"


Now let's go to the `serve` folder and create a virtual 
environment

```bash
pipenv --python=3.10
pipenv install scikit-learn==1.2.2 mlflow==1.29.0 boto3 flask gunicorn
```

Create a simple flask app (see [`serve.py`](serve/serve.py))


Run it:

```bash
echo ${MODEL_METADATA} | jq

export MODEL_VERSION=$(echo ${MODEL_METADATA} | jq -r ".run_id")
export MODEL_URI=$(echo ${MODEL_METADATA} | jq -r ".source")

pipenv run python serve.py
```

Test it:

```bash
REQUEST='{
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}'
URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL}
```

Now package the model with Docker and deploy it
(outside of the scope for this tutorial).



## Part 5: Monitoring


## Links

### Also see

* https://github.com/DataTalksClub/mlops-zoomcamp
* https://github.com/alexeygrigorev/lightweight-mlops-zoomcamp
* https://github.com/alexeygrigorev/ml-observability-workshop