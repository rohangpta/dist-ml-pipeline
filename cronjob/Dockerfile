FROM python:3.6-buster

RUN pip3 install kubernetes

WORKDIR /cronjob/

COPY ./cronjob /cronjob/
COPY ./k8s/training.yaml /cronjob/

CMD ["python3", "-u", "ptjob_sdk.py"]