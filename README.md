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


Note: used as final project for CIS 188 (https://cis188.org/)
