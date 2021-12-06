from kubernetes import client, config
import yaml

config.load_kube_config()

# Use the k8s Python SDK to simulate a cronjob for the PyTorchJob CRD

api = client.CustomObjectsApi()
response = api.list_cluster_custom_object(
    group="kubeflow.org", version="v1", plural="pytorchjobs", watch=False
)

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
    print(resp)

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
