FROM python:3.7-alpine

COPY bot_scripts/config.py /bots/
COPY bot_scripts/tweet.py /bots/
COPY bot_scripts/count_logger.json /bots/
COPY requirements.txt /tmp
RUN apk add curl
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /bots
CMD ["python3", "tweet.py"]
