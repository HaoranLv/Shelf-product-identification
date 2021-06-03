# FROM silverlogic/python3.6
# FROM tensorflow/tensorflow:1.14.0-gpu-py3
FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN mkdir -p /opt/ml/model

# COPY package/ /opt/ml/code/package/

# COPY serve.py /opt/ml/model/code/

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install --upgrade pip setuptools wheel
RUN pip install networkx==2.3 flask gevent gunicorn boto3 opencv-python==4.4.0.40 matplotlib -i https://opentuna.cn/pypi/web/simple
RUN pip install Keras==2.2.5 keras-retinanet==0.5.1 numpy==1.18 tqdm==4.50.2 pandas==0.23.4 h5py==2.10.0
### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN apt update
RUN echo y | apt install libgl1-mesa-glx

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

# Set up the program in the image
COPY ./ /opt/program/
WORKDIR /opt/program

ENTRYPOINT ["python", "serve.py"]

