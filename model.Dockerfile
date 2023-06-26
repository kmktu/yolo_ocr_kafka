# Nvidia cuda used
FROM nvidia/cuda:11.6.0-devel-ubuntu18.04
RUN apt-get update -y
#RUN apt install -y python3.8
RUN apt install -y python3-pip python3-dev build-essential git vim libgl1-mesa-glx pkg-config locales

# COPY folder
#COPY test /yolo_ocr/test
COPY weights /yolo_ocr/weights
COPY yolov7 /yolo_ocr/yolov7
#COPY Yolov7_StrongSORT_OSNet /yolo_ocr/Yolov7_StrongSORT_OSNet
COPY function.py /yolo_ocr/function.py
COPY kafka_producer.py /yolo_ocr/kafka_producer.py
COPY lp_detection_model.py /yolo_ocr/lp_detection_model.py
COPY requirements_model.txt /yolo_ocr/requirements_model.txt
#COPY docker-compose.yml /yolo_ocr/docker-compose.yml

RUN pip3 install -U pip setuptools wheel
RUN pip3 install -r /yolo_ocr/requirements_model.txt

# RTX 3090 version
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# locale
#RUN apt-get install -y locales
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8
WORKDIR /yolo_ocr
#ENTRYPOINT ["set", "+H"]
#ENTRYPOINT ["python3", "-u", "lp_detection_model_init.py"]