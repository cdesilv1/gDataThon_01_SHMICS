FROM python:3.7-alpine

COPY bot_scripts/config.py /bots/
COPY bot_scripts/tweet.py /bots/
COPY bot_scripts/count_logger.json /bots/
COPY requirements.txt /tmp
RUN apk add curl
RUN apk add pkgconfig
RUN apk add --no-cache --virtual .build-deps \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    glib-dev \
    gobject-introspection-dev \
    cairo-dev \
    cairo \
    cairo-tools
RUN pip3 install -r /tmp/requirements.txt
RUN apk del .build-deps

RUN wget https://github.com/tanaikech/goodls/releases/download/v1.2.7/goodls_linux_amd64
RUN chmod +x ./goodls_linux_amd64

# REPLACE WITH GENERATED PROBIDEN JSON url
RUN ./goodls_linux_amd64 -f probiden.json -u https://drive.google.com/file/d/1ErGTyGntUIM6OnMgW1f18WU51fY7TN1H/view?usp=sharing

# REPLACE WITH GENERATED PROTRUMP JSON url
RUN ./goodls_linux_amd64 -f protrump.json -u https://drive.google.com/file/d/1aQuMi6hLnpDO3Y90eQgVO3fNGtMNKFg8/view?usp=sharing

# moving generated files to workdir
RUN mv probiden.json /bots/probiden.json
RUN mv protrump.json /bots/protrump.json

RUN rm goodls_linux_amd64

WORKDIR /bots
CMD ["python3", "tweet.py"]
