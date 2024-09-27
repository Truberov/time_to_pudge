FROM python:3.10-buster

WORKDIR /code

RUN pip install poetry

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

COPY ./rutube_bot ./rutube_bot

CMD python3 -m rutube_bot
