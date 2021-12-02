FROM python:3.9-buster

ENV POETRY_VERSION=1.0.8

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app/

RUN mkdir /app/models/

COPY ./app/pyproject.toml /app/
COPY ./app/poetry.lock /app/

RUN poetry install

COPY ./models/ /app/models/
COPY ./app/ /app/

CMD ["poetry", "run", "uvicorn", "serve:app", "--host", "0.0.0.0"]