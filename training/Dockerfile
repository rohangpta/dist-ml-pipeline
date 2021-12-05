FROM python:3.9-buster

ENV POETRY_VERSION=1.1.0

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /training/

RUN mkdir /training/models/

COPY ./training/pyproject.toml /training/
COPY ./training/poetry.lock /training/

RUN poetry install

COPY ./models/ /training/models/
COPY ./training/ /training/

ENTRYPOINT ["poetry", "run", "python3", "-u", "mnist.py"]