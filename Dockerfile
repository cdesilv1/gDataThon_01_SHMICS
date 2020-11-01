FROM python:3.7-alpine

# SETUP RCLONE 
# https://rclone.org/install/
# https://rclone.org/remote_setup/

COPY bot_scripts/config.py /bots/
COPY bot_scripts/tweet.py /bots/
COPY bot_scripts/count_logger.json /bots/
COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /bots
CMD ["python3", "tweet.py"]
