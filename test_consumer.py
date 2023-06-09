from kafka import KafkaConsumer
from json import loads
import json
import requests
import datetime
import time
import string

def main():
    consumer = KafkaConsumer(
        'ocr_result',
        bootstrap_servers=['localhost:29092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        consumer_timeout_ms=1000
    )

    print(consumer.topics())
    print("[begin] get consumer list")
    while True:
        value = get_message(consumer)

        if value:
            post_send_m(value)
            # print("-" * 10)
            # print("value : ", value)
            # print("-" * 10)


def post_send_m(value):
    if value is not None:
        detect_time = value['Detect_Time']
        ocr_last_4 = value['Ocr_Result_4']
        ocr_all_number = value['Ocr_Result_All']
        truck_count = value['Truck_Count']
        camera_id = value['Camera_id']

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
        camera_guid = ""
        scraparea_cd = ""
        if camera_id == '111111111':
            camera_guid = cameraGuid_list[0]
            scraparea_cd = scrapAreaCd_list[0]

        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
                       "date": ""}
        new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}

        result_dict["rcnCarNumb"] = ocr_last_4
        new_result_dict["rcnCarNumb"] = ocr_last_4
        scheduled_plate = find_truck_schedule(ocr_last_4, truck_schedule_url_list[2])

        if scheduled_plate == "":
            new_result_dict["rcnCarNumb"] = ocr_last_4
            new_result_dict["procYn"] = 'N'
            post_send(new_result_dict, post_dest_url_list[5], scraparea_cd, camera_guid)
        else:
            print("START")
            new_result_dict["rcnCarNumb"] = ocr_last_4
            new_result_dict["procYn"] = 'Y'
            new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
            new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
            # new_post_send(new_result_dict, post_dest_url_list[1])
            # new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
            post_send(new_result_dict, post_dest_url_list[5], scraparea_cd, camera_guid)

        print("-" * 10)
        print("[Detect Time] : ", detect_time)
        print("[Ocr Result last 4] : ", ocr_last_4)
        print("[Ocr Result All Number] : ", ocr_all_number)
        print("[Truck Count] : ", truck_count)
        print("[Camera ID] : ", camera_id)
        print("-" * 10)
    else:
        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        new_result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": "",
                           "cameraGuid": "", "scrapAreaCd": ""}



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
            #if compareCount >= 3:
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

if __name__ == '__main__':
    main()