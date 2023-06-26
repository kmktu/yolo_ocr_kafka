# Nvidia cuda used
FROM ubuntu:18.04
RUN apt-get update -y
RUN apt install -y python3.8 python3-pip python3-dev build-essential git vim libgl1-mesa-glx pkg-config

COPY consumer_post.py /yolo_ocr/consumer_post.py
COPY requirememts_consumer.txt /yolo_ocr/requirememts_consumer.txt
RUN pip3 install -r /yolo_ocr/requirememts_consumer.txt

#ENTRYPOINT ["tail", "-f", "/dev/null"]
WORKDIR /yolo_ocr
#ENTRYPOINT ["python3", "-u", "consumer_post.py"]