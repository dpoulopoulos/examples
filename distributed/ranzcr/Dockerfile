FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime 

RUN pip install tensorboardX pandas scikit-learn

WORKDIR /opt
COPY ranzcr /opt/ranzcr

RUN  chgrp -R 0 /opt/ranzcr \
     && chmod -R g+rwX /opt/ranzcr

ENTRYPOINT ["python", "/opt/ranzcr/ranzrc.py"]