# Still didnt understand anything
FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    protobuf-compiler

COPY . /app

RUN protoc object_detection/protos/*.proto --python_out=.
RUN python -m pip install -U pip
RUN python -m pip install .
RUN python -m pip install streamlit

EXPOSE 8501

CMD streamlit run --server.port 8501 --server.enableCORS false app.py