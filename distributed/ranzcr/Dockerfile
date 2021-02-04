FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime 

COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /opt
COPY ranzcr /opt/ranzcr

RUN  chgrp -R 0 /opt/ranzcr \
     && chmod -R g+rwX /opt/ranzcr

ENTRYPOINT ["python", "/opt/ranzcr/main.py"]