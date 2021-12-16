# End to End Machine Learning Development

Develop an end-to-end Machine Learning pipeline with a training job, sample model definition, and prediction service.

Integrate support for distributed workloads using PyTorchJob, Continuous Training using CronJob (implemented using the kubernetes Python SDK) and Continuous Monitoring using Fluent Bit as a DaemonSet, hooked to S3.

Follow conventional DevOps & MLOps practices including separating training and prediction, containerisation & deployment on Kubernetes. Further support for declarative development using a `kind_config.yaml` file. 

## Key features:

- Model-agnostic training framework
- Declarative development (in Kind)
- ML-at-scale via distributed PyTorch jobs
- Continuous Training via CronJob to retrigger
- Continuous Monitoring via Fluent Bit and S3

## Setup Instructions

After cloning the repository, build the images via the provided `docker-compose.yml` file. If running in Kind, create cluster, load images, and apply manifests. Ensure `k8s/cronjob.yaml` has the correct schedule and S3 credentials are provided via Kubernetes secrets. 

Plug-and-play with different models in `models/`, and tune hyparameters in `k8s/training.yaml`. Customise number of workers, GPU usage & training backend in `k8s/training.yaml`.

Run standalone jobs during model development using `k create job --from=cronjob/cron-ptjob cron-ptjob`.


Note: used as final project for CIS 188 (https://cis188.org/)
