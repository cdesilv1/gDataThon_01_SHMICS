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
    cairo-tools \
    py3-cairo-dev
RUN pip3 install -r /tmp/requirements.txt

# probiden
ARG fileid="1ErGTyGntUIM6OnMgW1f18WU51fY7TN1H"
ARG filename="probiden.json"
RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
RUN curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# protrump
ARG fileid="1aQuMi6hLnpDO3Y90eQgVO3fNGtMNKFg8"
ARG filename="protrump.json"
RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
RUN curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

# remove build deps
RUN apk del .build-deps

# moving generated files to workdir
RUN mv probiden.json /bots/probiden.json
RUN mv protrump.json /bots/protrump.json

WORKDIR /bots
CMD ["python3", "tweet.py"]
