FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    git

WORKDIR /app

COPY ./requirements/requirements_server.txt /app/

RUN pip install -r requirements_server.txt

COPY ./src/server/api_server.py /app/

CMD ["/bin/bash"]