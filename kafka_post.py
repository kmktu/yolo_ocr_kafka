from kafka import KafkaConsumer
from json import loads
import time
import requests
import json

consumer = KafkaConsumer(
    'ocr_result',
    bootstrap_servers=['1.209.33.170:29092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    consumer_timeout_ms=1000
)
cameraId_list = [111111111, 222222222, 333333333]
cameraGuid_list = ['5bbe27d6-690d-4e27-b153-e60d207164a6',
                   'FF80F014-EDFC-499B-BDD5-A1997909DD3B',
                   '173BC97B-1780-41FC-9671-E43C2ED7C158']
# scrapAreaCd_list = ['2100E1', '1200A1', '1200B2'] #하차구역
scrapAreaCd_list = ['2100E1'] #하차구역
post_dest_url_list = ["http://ims.yksteel.co.kr:90/WebServer/AiResult",
                      "http://192.168.70.43:8080/ai/receiveVehicleInfo",
                      "http://192.168.20.216:8080/ai/receiveVehicleInfo",
                      "http://1.209.33.170:8099/get",
                      "http://imsns.idaehan.com:8080/ai/receiveVehicleInfo",
                      "https://imsns.idaehan.com:8443/ai/receiveVehicleInfo"]
truck_schedule_url_list = ["http://ims.yksteel.co.kr:90/WebServer/MstrWait",
                           "http://192.168.70.30:8080/tms/searchIncomongVehicleList.do",
                           "https://wss.idaehan.com:8443/tms/searchIncomongVehicleList.do"]

def kafka_consumer(consumer):
    value = None
    for message in consumer:
        topic = message.topic
        partition = message.partition
        offset = message.offset
        key = message.key
        value = loads(message.value.decode('utf-8'))
        # data = [value['Detect_Time'], value['Ocr_Result_4'], value['Ocr_Result_All'], value['Truck_Count']]
    print("[begin] get consumer list : ", consumer.topics())

    return value

def new_post_send(data, dest_url): #tmsOdrNo(배차정보) , vhclNo(차량정보), cameraGuid(카메라GUID), scrapAreaCd(하차구역)
    try:
        data["cameraGuid"] = cameraGuid_list[0]
        data["scrapAreaCd"] = scrapAreaCd_list[0]
        result = {"Result": [data]}
        json_data = json.dumps(result)
        headers = {'Content-Type': 'application/json'}
        r = requests.post(url=dest_url, data=json_data, verify=False, headers=headers, timeout=1)
        file_write(dest_url + ", " + str(json_data))
        print("POST!! : ", dest_url + ", " + str(json_data))
        r.close()
    except Exception as e:
        print("================= POST Send Exception =================")
        print(e)
        file_write("[error]"+str(e))
        # r.close()
        print("=======================================================")

def file_write(data):
    try:
        print("-------------------------print-------------------------------")
        print("[log write]"+data)
        f = open("ocr_log.txt", "a")
        f.write("\n" + str(data))
        f.close()
    except Exception as e:
        print(e)

def get_last_number(plate):
    try:
        last_num = plate[-4:]
    except Exception as e:
        return ""
    return last_num

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

def find_plate():
    while True:
        value = kafka_consumer(consumer)
        if value is None:
            print("Value is None")
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            new_result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "",
                               "vhclNo": "", "cameraGuid": "", "scrapAreaCd": ""}
            # new_post_send(new_result_dict, None, post_dest_url_list[1])
            # new_post_send(new_result_dict, post_dest_url_list[1])
            # new_post_send(new_result_dict, post_dest_url_list[5])
            new_post_send(new_result_dict, post_dest_url_list[5])
        else:
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
                           "date": ""}
            new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}
            ocr_detect_time = value['Detect_Time']
            ocr_result_last_4 = value['Ocr_Result_4']
            ocr_result_all = value['Ocr_Result_All']
            ocr_truck_count = value['Truck_Count']
            print("-" * 10)
            print("[Detect Time] : ", ocr_detect_time)
            print("[Ocr Result last 4] : ", ocr_result_last_4)
            print("[Ocr Result All Number] : ", ocr_result_all)
            print("[Truck Count] : ", ocr_truck_count)
            print("-" * 10)

            # camera_id = []

            result_dict["rcnCarNumb"] = ocr_result_last_4
            new_result_dict["rcnCarNumb"] = ocr_result_last_4

            scheduled_plate = find_truck_schedule(ocr_result_last_4, truck_schedule_url_list[2])
            print("scpl : ", scheduled_plate)

            if scheduled_plate == "":
                pass
            else:
                print("START")
                new_result_dict["rcnCarNumb"] = ocr_result_last_4
                new_result_dict["procYn"] = 'Y'
                new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
                new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
                # new_post_send(new_result_dict, post_dest_url_list[1])
                # new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
                new_post_send(new_result_dict, post_dest_url_list[5])

            # print("openalpr: ", camera_id, ", best_num: ", best_plate, ", get_num: ", new_result_dict["vhclNo"])
            # file_write(
            #     "openalpr: " + str(camera_id) + ", best_num: " + str(best_plate) + ", get_num: " + new_result_dict[
            #         "vhclNo"])
            # file_write("---------------------------------------------")
            # print("---------------------------------------------------")

if __name__ == '__main__':
    find_plate()