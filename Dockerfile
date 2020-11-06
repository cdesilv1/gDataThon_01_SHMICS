FROM python:3.7-alpine

COPY bot_scripts/config.py /bots/
COPY bot_scripts/tweet.py /bots/
COPY bot_scripts/count_logger.json /bots/
COPY requirements.txt /tmp
RUN apk add curl
RUN pip3 install -r /tmp/requirements.txt

RUN wget https://github.com/tanaikech/goodls/releases/download/v1.2.7/goodls_linux_amd64
RUN chmod +x ./goodls_linux_amd64

# REPLACE WITH GENERATED PROBIDEN JSON url
RUN ./goodls_linux_amd64 -f probiden.json -u https://drive.google.com/file/d/1K_zaeE48eD-C1KEFReGQ3Bvr8vs6LFHe/view?usp=sharing

# REPLACE WITH GENERATED PROTRUMP JSON url
RUN ./goodls_linux_amd64 -f protrump.json -u https://drive.google.com/file/d/1K_zaeE48eD-C1KEFReGQ3Bvr8vs6LFHe/view?usp=sharing

# moving generated files to workdir
RUN mv probiden.json /bots/probiden.json
RUN mv protrump.json /bots/protrump.json

RUN rm goodls_linux_amd64

WORKDIR /bots
CMD ["python3", "tweet.py"]
