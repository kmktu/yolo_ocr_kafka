from kafka import KafkaConsumer
from json import loads
import json
import requests
import datetime
import time
import string
import threading
import argparse

# class post_t (threading.Thread):
#     def __init__(self, consumer, receive_topic_name):
#         super().__init__()
#         self.consumer = consumer
#         self.receive_topic_name = receive_topic_name
#
#     def run(self):
#         print("START")
#         while True:
#             value = self.get_message(self.consumer)
#             print("value : ", value)
#             if value:
#                 self.post_send_m(value)

def main(bs, topic):
    stream_consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[bs],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        consumer_timeout_ms=1000
    )

    while True:
        value = get_message(stream_consumer)
        if value:
            post_send_m(value)
    # stream_1_receive_topic_name = "stream_1_result"
    # stream_2_receive_topic_name = "stream_2_result"
    # stream_3_receive_topic_name = "stream_3_result"
    #
    # # kafka 컨슈머 연결
    # stream_1_consumer = KafkaConsumer(
    #     stream_1_receive_topic_name,
    #     bootstrap_servers=['kafka2:9093'],
    #     auto_offset_reset='latest',
    #     enable_auto_commit=True,
    #     consumer_timeout_ms=1000
    # )
    # stream_2_consumer = KafkaConsumer(
    #     stream_2_receive_topic_name,
    #     bootstrap_servers=['kafka2:9093'],
    #     auto_offset_reset='latest',
    #     enable_auto_commit=True,
    #     consumer_timeout_ms=1000
    # )
    # stream_3_consumer = KafkaConsumer(
    #     stream_3_receive_topic_name,
    #     bootstrap_servers=['kafka2:9093'],
    #     auto_offset_reset='latest',
    #     enable_auto_commit=True,
    #     consumer_timeout_ms=1000
    # )
    #
    # print("[begin] get consumer list")
    #
    # stream_1_post_t = post_t(stream_1_consumer, stream_1_receive_topic_name)
    # stream_2_post_t = post_t(stream_2_consumer, stream_2_receive_topic_name)
    # stream_3_post_t = post_t(stream_3_consumer, stream_3_receive_topic_name)
    #
    # stream_1_post_t.start()
    # stream_2_post_t.start()
    # stream_3_post_t.start()

def get_message(consumer):
    value = None
    if consumer:
        for message in consumer:
            topic = message.topic
            partition = message.partition
            offset = message.offset
            key = message.key
            value = loads(message.value.decode('utf-8'))
    return value

def post_send_m(value):
    post_dest_url_list = ["http://ims.yksteel.co.kr:90/WebServer/AiResult",
                          "http://192.168.70.43:8080/ai/receiveVehicleInfo",
                          "http://192.168.20.216:8080/ai/receiveVehicleInfo",
                          "http://1.209.33.170:8099/get",
                          "http://imsns.idaehan.com:8080/ai/receiveVehicleInfo",
                          "https://imsns.idaehan.com:8443/ai/receiveVehicleInfo"]
    truck_schedule_url_list = ["http://ims.yksteel.co.kr:90/WebServer/MstrWait",
                               "http://192.168.70.30:8080/tms/searchIncomongVehicleList.do",
                               "https://wss.idaehan.com:8443/tms/searchIncomongVehicleList.do"]
    cameraGuid_list = ['5c2de1a4-ef27-4780-8759-9ee9830b83f2',
                       'FF80F014-EDFC-499B-BDD5-A1997909DD3B',
                       '173BC97B-1780-41FC-9671-E43C2ED7C158']
    scrapAreaCd_list = ['2100E1', '1200A1', '1200B2']

    if value is not None:
        detect_time = value['Detect_Time']
        ocr_last_4 = value['Ocr_Result_4']
        ocr_all_number = value['Ocr_Result_All']
        truck_count = value['Truck_Count']
        camera_id = value['Camera_id']
        # ocr_result = value['ocr_result']
        # camera_id = value['camera_id']
        camera_guid = ""
        scraparea_cd = ""

        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
                       "date": ""}
        new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}

        # 카메라 아이디는 kafka 메시지를 통해 들어옴
        # cameraGuid, scrapAreaCd 는 동연에서 정해준 것 사용
        if camera_id == '1':
            camera_guid = cameraGuid_list[0]
            scraparea_cd = scrapAreaCd_list[0]
        elif camera_id == '2':
            camera_guid = cameraGuid_list[1]
            scraparea_cd = scrapAreaCd_list[1]
        elif camera_id == '3':
            camera_guid = cameraGuid_list[2]
            scraparea_cd = scrapAreaCd_list[2]

        if ocr_last_4 == "":
            post_send(new_result_dict, post_dest_url_list[5], scraparea_cd, camera_guid)
        else:
            scheduled_plate = find_truck_schedule(ocr_last_4, truck_schedule_url_list[2])

            if scheduled_plate == "":
                new_result_dict["rcnCarNumb"] = ocr_last_4
                new_result_dict["procYn"] = 'N'
                post_send(new_result_dict, post_dest_url_list[5], scraparea_cd, camera_guid)
            else:
                new_result_dict["rcnCarNumb"] = ocr_last_4
                new_result_dict["procYn"] = 'Y'
                new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
                new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
                post_send(new_result_dict, post_dest_url_list[5], scraparea_cd, camera_guid)


# POST 통신 부분
def post_send(data, dest_url, scraparea_cd, camera_guid):
    try:
        data["cameraGuid"] = camera_guid
        data["scrapAreaCd"] = scraparea_cd
        result = {"Result": [data]}
        json_data = json.dumps(result)
        headers = {'Content-Type': 'application/json'}
        r = requests.post(url=dest_url, data=json_data, verify=False, headers=headers, timeout=1)
        # file_write(dest_url + ", " + str(json_data))
        print("POST!! : ", dest_url + ", " + str(json_data))
        r.close()
    except Exception as e:
        print("================= POST Send Exception =================")
        print(e)
        print("=======================================================")

# 트럭 들어왔는지 확인하는 스케줄러
def find_truck_schedule(plate, url):
    r = requests.post(url)
    print("********************* truck schedule data *********************")
    # print("[GET DATA]", r.content)
    json_data = json.loads(r.content)
    parsed_arr = json_data["data"]
    if parsed_arr != None:
        # origin_4num = get_last_number(plate)
        for data in parsed_arr:
            car_num = data['vhclNo']
            schedule_4num = get_last_number(car_num)
            compareCount = 0
            origin_4num_arr = list(map(str, plate))
            car_num_arr = list(map(str, schedule_4num))
            for i in range(4):
                if (origin_4num_arr[i] == car_num_arr[i]):
                    compareCount += 1
            # change number of match number 4
            # if compareCount >= 3:
            if compareCount >= 4:
                print("*************************************************************", compareCount * 25)
                return data
    return ""

def get_last_number(plate):
    try:
        last_num = plate[-4:]
    except Exception as e:
        return ""
    return last_num

def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--bootstrap_server', default=None, help="use bootstrap_server")
    parser.add_argument('-t', '--topic', default=None, help="use topic")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = input_parser()
    bs = args.bootstrap_server
    topic = args.topic
    main(bs, topic)