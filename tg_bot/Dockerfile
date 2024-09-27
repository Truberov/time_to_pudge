FROM python:3.10-buster

WORKDIR /code

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./tg_bot ./tg_bot

CMD python3 -m tg_bot
