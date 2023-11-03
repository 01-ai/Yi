ARG REGISTRY="nvcr.io"
ARG CUDA_VERSION="11.8.0"
FROM ${REGISTRY}/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

ARG MAX_JOBS=8

RUN apt update \
  && apt install -y python3-pip python3-packaging git ninja-build \
  && pip3 install -U pip \
  && ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /yi

RUN pip3 install torch==2.0.1 \
  && pip3 install flash-attn==2.3.3  --no-build-isolation

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
