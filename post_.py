import base64
import time
import beanstalkc
import json
import requests
import os
import datetime

COUNT_TRUCK_TUBE_NAME = 'truck_count'
camera_name = ['E001',  '녹산 고철 A동', '녹산 고철 B동']
#cameraId_list = [111111111, 222222222, 333333333]
cameraId_list = [222222222, 333333333]
cameraGuid_list = ['5bbe27d6-690d-4e27-b153-e60d207164a6', 'FF80F014-EDFC-499B-BDD5-A1997909DD3B', '173BC97B-1780-41FC-9671-E43C2ED7C158']
#scrapAreaCd_list = ['2100E1', '1200A1', '1200B2'] #하차구역
scrapAreaCd_list = ['1200A1', '1200B2'] #하차구역
post_dest_url_list = ["http://ims.yksteel.co.kr:90/WebServer/AiResult", "http://192.168.70.43:8080/ai/receiveVehicleInfo", "http://192.168.20.216:8080/ai/receiveVehicleInfo", "http://1.209.33.170:8099/get", "http://imsns.idaehan.com:8080/ai/receiveVehicleInfo", "https://imsns.idaehan.com:8443/ai/receiveVehicleInfo"]
truck_schedule_url_list = ["http://ims.yksteel.co.kr:90/WebServer/MstrWait", "http://192.168.70.30:8080/tms/searchIncomongVehicleList.do", "https://wss.idaehan.com:8443/tms/searchIncomongVehicleList.do"]

def new_post_send(data, camera_id, dest_url): #tmsOdrNo(배차정보) , vhclNo(차량정보), cameraGuid(카메라GUID), scrapAreaCd(하차구역)
    for idx, val in enumerate(cameraId_list):
        if camera_id == val:
            data["cameraGuid"] = cameraGuid_list[idx]
            data["scrapAreaCd"] = scrapAreaCd_list[idx]
    try:
        if camera_id == cameraId_list[0]:
            count = flush_and_get_last_queue(COUNT_TRUCK_TUBE_NAME)
            num_count = int(float(count))
            result = {"Result": [data], "Count": num_count}
        else:
            result = {"Result": [data]}
        json_data = json.dumps(result)
        headers = {'Content-Type': 'application/json'}
        r = requests.post(url=dest_url, data=json_data, verify=False, headers=headers, timeout=1)
        # file_write(dest_url + ", " + str(json_data))
        day_log_write("[success]" + dest_url + ", " + str(json_data))
        print(dest_url + ", " + str(json_data))
        r.close()
    except Exception as e:
        print("================= POST Send Exception =================")
        print(e)
        # file_write("[error]"+str(e))
        day_log_write("[error]" + str(e))
        # r.close()
        print("=======================================================")

def day_log_write(data):
    directory = "alpr_log"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
    detect_time = time.strftime('%Y-%m-%d')
    try:
        file_name = directory + "/alpr_log_" + detect_time + ".txt"
        if os.path.isfile(file_name):
            f = open(file_name, "a")
            f.write("\n" + str(data))
            f.close()
        else:
            f = open(file_name, "w")
            f.write("\n" + str(data))
            f.close()

    except Exception as e:
        print(e)


def file_write(data):
    try:
        print("-------------------------print-------------------------------")
        print("[log write]"+data)
        f = open("alpr_log.txt", "a")
        f.write("\n" + str(data))
        f.close()
    except Exception as e:
        print(e)


def err_log_file_write(data):
    try:
        # print("-------------------------print-------------------------------")
        # print("[error log write]"+data)
        f = open("err_log.txt", "a")
        f.write("\n" + str(data))
        f.close()
    except Exception as e:
        print(e)


def flush_queue():
    beanstalk = beanstalkc.Connection(host='localhost', port=11300)
    TUBE_NAME = 'alprd'
    beanstalk.watch(TUBE_NAME)
    run = True
    while run:
        job = beanstalk.reserve(timeout=0)
        if job is None:
            run = False
        else:
            job.delete()


def flush_and_get_last_queue(tube_name):
    beanstalk = beanstalkc.Connection(host='localhost', port=11300)
    beanstalk.watch(tube_name)
    last_value = ""
    run = True
    while run:
        job = beanstalk.reserve(timeout=0)
        if job is None:
            run = False
        else:
            last_value = job.body
            job.delete()
    return last_value


def find_plate():
    beanstalk = beanstalkc.Connection(host='localhost', port=11300)
    print(beanstalk.tubes())
    TUBE_NAME = 'alprd'
    beanstalk.watch(TUBE_NAME)

    # Loop forever
    while True:
        job = beanstalk.reserve(timeout=3.0)
        if job is None:
            print("----------------------------------------------")
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            new_result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": "", "cameraGuid": "", "scrapAreaCd": ""}
            # new_post_send(new_result_dict, None, post_dest_url_list[1])
            # new_post_send(new_result_dict, None, post_dest_url_list[2])
            new_post_send(new_result_dict, None, post_dest_url_list[5])
        else:
            # print(job.body)
            local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "", "date": ""}
            new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}
            plates_info = json.loads(job.body)
            if plates_info["data_type"] == "alpr_group" or plates_info["data_type"] == "alpr_results":
                # print(job.body)
                camera_id = plates_info["camera_id"]
                best_plate = plates_info["best_plate"]["plate"]
                #plate_crop_jpeg = plates_info["best_plate"]["plate_crop_jpeg"]
                #file_img_write(camera_id, best_plate, plate_crop_jpeg)
                result_dict["rcnCarNumb"] = best_plate
                new_result_dict["rcnCarNumb"] = best_plate
                #changed truck_schedule_url_list 1 -> 2 (https url)
                scheduled_plate = find_truck_schedule(best_plate, truck_schedule_url_list[2])
                if scheduled_plate == "":
                    pass
                else:
                    new_result_dict["rcnCarNumb"] = best_plate
                    new_result_dict["procYn"] = 'Y'
                    new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
                    new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
                    # new_post_send(new_result_dict, camera_id, post_dest_url_list[1])
                    #new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
                    new_post_send(new_result_dict, camera_id, post_dest_url_list[5])
                print("openalpr: ", camera_id, ", best_num: ", best_plate, ", get_num: ", new_result_dict["vhclNo"])
                # file_write("openalpr: "+ str(camera_id)+", best_num: "+str(best_plate)+", get_num: "+new_result_dict["vhclNo"])
                day_log_write("openalpr: "+ str(camera_id)+", best_num: "+str(best_plate)+", get_num: "+new_result_dict["vhclNo"])
                # file_write("---------------------------------------------")
                day_log_write("-"*20)
                print("---------------------------------------------------")
            else:
                pass
            job.delete()

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
        origin_4num = get_last_number(plate)
        for data in parsed_arr:
            car_num = data['vhclNo']
            schedule_4num = get_last_number(car_num)
            compareCount = 0
            origin_4num_arr = list(map(str, origin_4num))
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

def file_img_write(camera_id, best_plate, plate_crop_jpeg):
    local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    date = local_time.split(" ")[0]
    dtime = local_time.split(" ")[1]
    dtime = dtime.replace(":", "")
    current_path = os.getcwd()
    try:
        # os.makedirs(current_path + "/img/" + date, exist_ok=True)
        os.makedirs(current_path + "\\img\\" + date, exist_ok=True)
    except Exception as e:
        print(e)
    try:
        # f = open(current_path + "/img/" + date + "/" + str(camera_id) + "_" + str(best_plate) + "_" + dtime + ".jpeg", 'wb')
        f = open(current_path + "\\img\\" + date + "\\" + str(camera_id) + "_" + str(best_plate) + "_" + dtime + ".jpeg", 'wb')
        str2 = base64.b64decode(plate_crop_jpeg)
        f.write(str2)
        f.close()

        # f.write(str(plate_crop_jpeg))
    except Exception as e:
        print(e)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    flush_queue()
    find_plate()


    # 카메라 111111111 의 스케쥴러 서버(http://ims.yksteel.co.kr:90/WebServer/MstrWait) 에서 파싱 데이터 변경 (230112)
    # def find_truck_schedule(plate, url):
    #     r = requests.post(url)
    #     print("********************* truck schedule data *********************")
    #     # print("[GET DATA]", r.content)
    #     check = True
    #     if (url == truck_schedule_url_list[0]):
    #         parsed_arr = json.loads(r.content)
    #         check = True
    #     else:
    #         json_data = json.loads(r.content)
    #         parsed_arr = json_data["data"]
    #         check = False
    #
    #     if parsed_arr != None:
    #         origin_4num = get_last_number(plate)
    #         for data in parsed_arr:
    #             if (check == True):
    #                 car_num = data['VEHICLE_NO']
    #             else:
    #                 car_num = data['vhclNo']
    #             schedule_4num = get_last_number(car_num)
    #             compareCount = 0
    #             origin_4num_arr = list(map(str, origin_4num))
    #             car_num_arr = list(map(str, schedule_4num))
    #             for i in range(4):
    #                 if (origin_4num_arr[i] == car_num_arr[i]):
    #                     compareCount += 1
    #             if compareCount >= 3:
    #                 print("*************************************************************", compareCount * 25)
    #                 return data
    #     return ""

    # def find_plate():
    #     beanstalk = beanstalkc.Connection(host='localhost', port=11300)
    #     print(beanstalk.tubes())
    #     TUBE_NAME = 'alprd'
    #     beanstalk.watch(TUBE_NAME)
    #
    #     # Loop forever
    #     while True:
    #         job = beanstalk.reserve(timeout=1.0)
    #         if job is None:
    #             print("----------------------------------------------")
    #             local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    #             result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
    #                            "date": "", "cameraGuid": "", "scrapAreaCd": ""}
    #             new_result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": "",
    #                                "cameraGuid": "", "scrapAreaCd": ""}
    #             new_post_send(result_dict, None, post_dest_url_list[0])
    #             new_post_send(new_result_dict, None, post_dest_url_list[1])
    #             # new_post_send(new_result_dict, None, post_dest_url_list[2])
    #         else:
    #             # print(job.body)
    #             local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    #             result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
    #                            "date": ""}
    #             new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}
    #             plates_info = json.loads(job.body)
    #             if plates_info["data_type"] == "alpr_group" or plates_info["data_type"] == "alpr_results":
    #                 # print(job.body)
    #                 camera_id = plates_info["camera_id"]
    #                 best_plate = plates_info["best_plate"]["plate"]
    #                 # plate_crop_jpeg = plates_info["best_plate"]["plate_crop_jpeg"]
    #                 # file_img_write(camera_id, best_plate, plate_crop_jpeg)
    #                 result_dict["rcnCarNumb"] = best_plate
    #                 new_result_dict["rcnCarNumb"] = best_plate
    #                 if camera_id == cameraId_list[0]:
    #                     scheduled_plate = find_truck_schedule(best_plate, truck_schedule_url_list[0])
    #                     if scheduled_plate == "":
    #                         pass
    #                     else:
    #                         result_dict["procYn"] = 'Y'
    #                         result_dict["carNumb"] = scheduled_plate["VEHICLE_NO"]  # vhclNo
    #                         result_dict["scaleNumb"] = scheduled_plate["DELIVERY_ID"]  # tmsOdrNo
    #                         result_dict["date"] = scheduled_plate["CREATION_DATE"]  # 없음
    #                         new_result_dict["rcnCarNumb"] = best_plate
    #                         new_result_dict["procYn"] = 'Y'
    #                         new_result_dict["vhclNo"] = scheduled_plate["VEHICLE_NO"]
    #                         new_result_dict["tmsOdrNo"] = scheduled_plate["DELIVERY_ID"]
    #                     # post_send(result_dict)
    #                     new_post_send(result_dict, camera_id, post_dest_url_list[0])
    #                     new_post_send(new_result_dict, camera_id, post_dest_url_list[1])
    #                     # new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
    #                 else:
    #                     scheduled_plate = find_truck_schedule(best_plate, truck_schedule_url_list[1])
    #                     if scheduled_plate == "":
    #                         pass
    #                     else:
    #                         new_result_dict["rcnCarNumb"] = best_plate
    #                         new_result_dict["procYn"] = 'Y'
    #                         new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
    #                         new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
    #                         new_post_send(new_result_dict, camera_id, post_dest_url_list[1])
    #                         # new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
    #                 print("openalpr: ", camera_id, ", best_num: ", best_plate, ", get_num: ", new_result_dict["vhclNo"])
    #                 file_write("openalpr: ", camera_id, ", best_num: ", best_plate, ", get_num: ",
    #                            new_result_dict["vhclNo"])
    #             else:
    #                 pass
    #             job.delete()
