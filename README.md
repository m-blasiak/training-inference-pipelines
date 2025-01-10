# Note
Normally the inference and training pipelines should be separated. 
I bundled them together into a single repo and created the docker-compose only for the sake of the exercise

# Building the container
In order to build the containers please run
```docker-compose build```

# Running the container
Please run the following command:
```docker-compose up```
This will:
- start a local mlflow server
- run the model-training pipeline that trains the model and exports it to MLFlow
- start a FastAPI based inference container

You should now be able to see a ``logdir`` folder. That folder will contain
mlflow artifacts and backend as well as a ``validation-data`` folder.

That folder contains JSON files with the validation dataset.
``x_val.json`` file is structured in a way that makes it easy to test the model
with postman/curl. 

# Calling the API
``127.0.0.1:8000/predict``

The FastAPI runs on a localhost at port 8000.
It has a POST ``/predict`` endpoint that takes a JSON body in the following format:
```
{
    "user_id": "123",
    "features": {
    "age": 52,
    "income": 105824,
    "employment_type": "part_time",
    "marital_status": "married",
    "time_spent_on_platform": 87.6188186,
    "number_of_sessions": 6,
    "fields_filled_percentage": 0.8534444659,
    "previous_year_filing": 0,
    "device_type": "desktop",
    "referral_source": "social_media_ad"
  }
}
```
Grab any of the data points from the X_val.json file. You can use the index as the user_id

Alternatively you can simply use curl
```
curl --location '127.0.0.1:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "user_id": "123",
    "features": {
    "age": 52,
    "income": 105824,
    "employment_type": "part_time",
    "marital_status": "married",
    "time_spent_on_platform": 87.6188186,
    "number_of_sessions": 6,
    "fields_filled_percentage": 0.8534444659,
    "previous_year_filing": 0,
    "device_type": "desktop",
    "referral_source": "social_media_ad"
  }
}'
```

# Accessing the model info in MLFlow
You should be able to view the training info on ``http://127.0.0.1:5000/``
The newly trained model will be registered as "taxfix_classifier" with a "champion" alias