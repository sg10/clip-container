FROM python:3.8.0-slim AS build

RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y python3-dev build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev libjpeg62 && \
    apt-get install -y wget git ca-certificates curl nginx python3-opencv


WORKDIR /build
RUN mkdir -p /opt/ml

COPY requirements.txt .
ENV PATH=/root/.local/bin:$PATH

RUN pip3 install --user --upgrade pip
RUN pip install --user cython
RUN pip3 install --user torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --user -r requirements.txt

COPY /code . 

FROM python:3.8.0-slim

RUN apt-get clean && \
    apt-get update -y && \
    apt-get install -y libjpeg62 ca-certificates nginx python3-opencv

ENV PATH="/opt/ml/code:/root/.local/bin:${PATH}"
RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

COPY --from=build /root/.local /root/.local
COPY --from=build /build/ .
COPY --from=build /opt/ml /opt/ml

# SageMaker will automatically run the serve script so we need to make
# sure it has execution permissions.
RUN chmod +x serve