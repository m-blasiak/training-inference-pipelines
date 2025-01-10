
## How the solution integrates into the Taxfix product workflow.
A churn prediction model such as this one can improve conversion rate and thus improve Taxfix's bottom line.
It can also help optimize marketing spend by ensuring that vouchers/promotions are targeted to users at high risk of churn.

The model can be integrated into the funnel and make a prediction after each session. 
When a user exits the app without completing their tax filing, the backend triggers a call to the churn model
The model can assigns a churn probability score to the user.
This score, along with other user data should be sent to a CRM system where it can be used to decide on interventions 
such as personalized push notifications, email reminders, or targeted voucher offers.


## Steps to deploy the ML solution to a cloud environment (AWS/GCP/Azure):
I would consider model-training and model-inference to be separate projects/repositories.
**I bundled them together with a docker-compose only for the sake of the exercise.**
Also, right now the model requires features to be passed by the client. Depending on the specific use case this may not
be the optimal solution. In case, the client cannot easily obtain the features, we can modify the code to retrieve
the features from a feature store.

### Model Training Pipeline
Looking at Taxfix's existing [tech stack](https://stackshare.io/taxfix/taxfix) I would suggest orchestrating
the model training pipeline via Airflow.
A DAG combining data collection & cleanup with the training code can be created.
The actual processing would happen on a K8s cluster. Using either K8sPodOperator or a GKEPodOperator. 
Required resources can be specified in a config file used by the DAG. 

CiCD can be created using GitHub Actions.
Ci should include a lint & pytest steps (It can leverage the pre-commit config from the repo).
CD step to Dev,Staging & Prod environments (with appropriate triggers) should involve building the container, 
pushing it to Artifact Registry and updating an Airflow variable that points to the image in AR.

### Inference pipeline
The easiest solution to deploy the Inference pipeline is to use Cloud Run. It could be deployed in minutes with
a little bit of Terraform code. However, depending on the scaling and latency requirements, K8s deployment 
might be more appropriate.

CI could be very similar to the Training Pipeline.
CD would also involve building the image and pushing it to an Artifact Registry. For Cloud Run deployment, the CD would
either run a gcloud command to update the service to use the newer image, or orchestrate an update of the image version
in the appropriate TF repo.


## Monitoring and retraining strategies to ensure model performance 
With Airflow orchestrating model training process, adding a champion-challenger job to manage auto-deployments is a good idea.
It can compare the newly trained and existing models using a validation dataset, assign MLFlow's alias to the champion
and re-start the inference service (to re-load the model).

Monitoring of the model training jobs can be done directly via Airflow UI & dedicated hooks that alert to pipeline failures.

The inference pipeline can be integrated with Grafana/Prometheus. This will allow us to easily monitor the containers and
detect any performance issues.

In order to effectively monitor model performance, we need to store model responses in a data warehouse. The code already 
includes a placeholder function for this. It can be developed to asynchronously stream the responses to a PubSub topic. 
From there it can be pushed  either to BigQuery (native integration) or SnowFlake (using ApacheBeam/DataFlow).

Once the data is in the data warehouse, a separate Airflow job can run a number of tests to detect issues such as data drift.
This can be achieved with tools such as Evidently AI

## Scalability considerations (e.g., load balancing, autoscaling).
The pipeline is likely to be CPU bound. For Cloud Run, we can set up scaling based on CPU utilization. 
For K8s Horizontal Pod Autoscaler can be used to dynamically adjust the number of pods based on resource usage.
Load balancing should be implemented using a managed solution like Google Cloud Load Balancer or 
Kubernetes Ingress to distribute traffic efficiently. 

Regular load testing with tools like Locust or k6 should be conducted to validate performance under peak conditions

