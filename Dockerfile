FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
COPY . /pkg
WORKDIR /pkg
