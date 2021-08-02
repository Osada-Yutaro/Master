FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update
RUN apt-get -y install git
RUN apt-get install -y libgl1-mesa-dev
RUN python -m pip install --no-cache-dir opencv-python
RUN git clone https://github.com/Osada-Yutaro/Master
WORKDIR Master
CMD make detection_train