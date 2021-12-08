from kubernetes import client, config
import yaml

config.load_incluster_config()

# Use the k8s Python SDK to simulate a CronJob for the PyTorchJob custom resource

# Specifically, delete existing PyTorchJob and create a new one with standard spec (copied from k8s folder)

api = client.CustomObjectsApi()
pt_body = yaml.safe_load(open("./training.yaml"))


print("Deleting old object...")
try:
    resp = api.delete_namespaced_custom_object(
        group="kubeflow.org",
        version="v1",
        plural="pytorchjobs",
        namespace="default",
        name="pt-job",
    )
    print("\n\n\n\n")

except Exception:
    print("Not Found")

resp = api.create_namespaced_custom_object(
    group="kubeflow.org",
    version="v1",
    plural="pytorchjobs",
    body=pt_body,
    namespace="default",
)

print("Creating object...")
