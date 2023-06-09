# Nvidia cuda used
FROM nvidia/cuda:11.6.0-devel-ubuntu18.04
RUN apt-get update -y
#RUN apt install -y python3.8
RUN apt install -y python3-pip python3-dev build-essential git vim libgl1-mesa-glx pkg-config locales
#RUN apt-get install -y python3-pip python3-dev build-essential git vim
#RUN apt-get install -y git
#RUN apt-get install -y vim

# opencv Error problem solution
#RUN apt-get -y install libgl1-mesa-glx
#RUN apt-get -y install pkg-config

# COPY folder
#COPY test /yolo_ocr/test
COPY weights /yolo_ocr/weights
COPY yolov7 /yolo_ocr/yolov7
COPY Yolov7_StrongSORT_OSNet /yolo_ocr/Yolov7_StrongSORT_OSNet
COPY function.py /yolo_ocr/function.py
COPY lp_detection_tracking.py /yolo_ocr/lp_detection_tracking.py
COPY main.py /yolo_ocr/main.py
COPY ocr_log.txt /yolo_ocr/ocr_log.txt
COPY requirements.txt /yolo_ocr/requirements.txt
COPY kafka_producer.py /yolo_ocr/kafka_producer.py
COPY docker-compose.yml /yolo_ocr/docker-compose.yml
#COPY kafka_producer_post.py /yolo_ocr/kafka_producer_post.py
COPY main_send_post.py /yolo_ocr/main_send_post.py
COPY consumer_post.py /yolo_ocr/consumer_post.py
COPY test_consumer.py /yolo_ocr/test_consumer.py
#COPY test_producer.py /yolo_ocr/test_producer.py
#COPY test/test6.mp4 test/test6.mp4

RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r /yolo_ocr/requirements.txt

# RTX 3090 version
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
#RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# locale
#RUN apt-get install -y locales
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8

# setup module
RUN git clone https://github.com/KaiyangZhou/deep-person-reid
WORKDIR /deep-person-reid
RUN python3 setup.py develop

# Keep Runing process (dedug)
ENTRYPOINT ["tail", "-f", "/dev/null"]
WORKDIR /yolo_ocr
#ENTRYPOINT ["set", "+H"]
#ENTRYPOINT ["python3", "-u", "kafka_producer_post.py", "-c" ,"rtsp://admin:admin13579!@220.95.111.220/profile2/media.smp"]