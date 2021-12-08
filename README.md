# End-To-End MLOps 

## CIS 188 Final Project

Develop an end-to-end Machine Learning pipeline with support for distributed workloads using PyTorchJob, continuous training using CronJob (implemented using the kubernetes Python SDK) and Continuous Monitoring using Fluent Bit as a DaemonSet, hooked to S3.

Follow conventional DevOps & MLOps practices including separating training and prediction, containerisation & deployment on K8s. Further support for declarative development using a `kind_config.yaml` file. 

Aim to supply a model agnostic training & testing framework to streamline development while also being production ready and offering retraining in the event of new data, model architecture, etc.

Author: Rohan Gupta