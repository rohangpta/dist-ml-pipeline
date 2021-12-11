# End-To-End MLOps 

## CIS 188 Final Project

Develop an end-to-end Machine Learning pipeline with support for distributed workloads using PyTorchJob, Continuous Training using CronJob (implemented using the kubernetes Python SDK) and Continuous Monitoring using Fluent Bit as a DaemonSet, hooked to S3.

Follow conventional DevOps & MLOps practices including separating training and prediction, containerisation & deployment on K8s. Further support for declarative development using a `kind_config.yaml` file. 

Key features:

- Model-agnostic training framework
- Declarative development
- ML-at-scale via distributed jobs
- Continuous Training via CronJob
- Continuous Monitoring via Fluent Bit and S3

Author: Rohan Gupta
