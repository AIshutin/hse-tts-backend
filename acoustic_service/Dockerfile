# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.05-py3
FROM ${FROM_IMAGE_NAME}
RUN apt update && apt upgrade -y
RUN mkdir app

ENV PYTHONPATH /app
COPY . /app
WORKDIR /app

ADD requirements.txt .
#RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
RUN pip install -r requirements.txt  --use-feature=2020-resolver
EXPOSE 5000
#ENTRYPOINT ['python3']
#CMD ['python3','app.py']
