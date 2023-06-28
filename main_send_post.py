# -*- coding:utf-8 -*-
import datetime
import os
import cv2
import time
from PIL import Image
import lp_detection_tracking
import numpy as np
import string
import argparse
import logging
from kafka import KafkaProducer
import traceback
# from json import dumps
import requests
import json

logging.basicConfig(level=logging.INFO)

def fps_calculator(cur_time, prev_time):
    sec = cur_time - prev_time
    prev_time = cur_time

    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    return fps_str, prev_time


def input_rtsp(camera_path, log_save_s, lp_conf, lp_img_save):
    prev_time = 0
    detection_yolo = lp_detection_tracking.lp_detection_tracking()
    detection_yolo.model_init()
    rtsp = cv2.VideoCapture(camera_path)

    """
    plate_detection_flag : yolo 모델에서 번호판이 탐지되었을 경우 알려주는 플래그
    plate_detection_tracking_flag : 설정된 프레임 후 모든 처리과정을 실행시키기 위한 플래그
    detection_frame_count :작은 크기의 번호판이 아닌 온전한 번호판을 탐지하기 위한 프레임 무시 번호, 설정된 프레임 동안 번호판만 탐지하기 위함
    del_id_list : 삭제할 ID 리스트
    detect_plate_id_dict : 탐지된 번호판 딕셔너리 {"id" : [frame count, 24frame chek]}
    plate_ocr_4word_dict : 번호판 마지막 4자리 점수 계산 딕셔너리 {ocr 4word : value}
    plate_ocr_all_number_dict : 번호판 전체 번호 점수 계산 딕셔너리 {ocr word : value}
    plate_ocr_5_char_dict : 뒤에서 다섯번째 '한글'을 나타내기 위한 딕셔너리 {ocr 5char : value}
    """
    plate_detection_tracking_flag = False
    detection_frame_count = 0
    plate_ocr_4word_dict = {}
    plate_ocr_all_number_dict = {}
    plate_ocr_5_char_dict = {}
    ocr_list = []
    count_24 = 0
    truck_count = 0

    while True:
        rtsp_ret, rtsp_img = rtsp.read()
        rtsp_cur_time = time.time()

        if rtsp_ret:
            if detection_frame_count == 60:
                plate_detection_tracking_flag = True

            rtsp_infer_img, plate_detection_flag, plate_id_list, truck_id_list, exit_flag, crop_plate, small_plate_detect = \
                detection_yolo.inference_img(rtsp_img, plate_detection_tracking_flag=plate_detection_tracking_flag)

            if plate_detection_flag and not small_plate_detect and not plate_detection_tracking_flag:
                detection_frame_count += 1
            else:
                detection_frame_count = 0

            if plate_detection_tracking_flag:
                if len(plate_id_list) > 0:
                    for index, value in enumerate(plate_id_list):
                        plate_id = value[0]
                        ocr_word = value[1]
                        all_plate_number = ocr_word.translate(str.maketrans('', '', string.punctuation))
                        all_plate_number = all_plate_number.replace(" ", "")
                        last_4_word = all_plate_number[-4:]

                        if len(all_plate_number) > 5:
                            if ocr_list and last_4_word == ocr_list[2]:
                                plate_detection_tracking_flag = False
                                detection_frame_count = 0
                                count_24 = 0
                                plate_ocr_all_number_dict.clear()
                                plate_ocr_4word_dict.clear()
                                plate_ocr_5_char_dict.clear()
                            else:
                                if last_4_word not in plate_ocr_4word_dict:
                                    plate_ocr_4word_dict[last_4_word] = 1
                                else:
                                    plate_ocr_4word_dict[last_4_word] += 1

                                if all_plate_number not in plate_ocr_all_number_dict:
                                    if all_plate_number[-5].isalpha():
                                        plate_ocr_5_char_dict[all_plate_number[-5]] = 1
                                    plate_ocr_all_number_dict[all_plate_number] = 1
                                else:
                                    if all_plate_number[-5].isalpha():
                                        plate_ocr_5_char_dict[all_plate_number[-5]] += 1
                                    plate_ocr_all_number_dict[all_plate_number] += 1

                                if count_24 == 24:
                                    # 추후 트럭의 움직임 유무 확인 시
                                    truck_T_S_F = max(exit_flag, key=exit_flag.get)
                                    exit_flag["True"] = 0
                                    exit_flag["Stop"] = 0
                                    exit_flag["False"] = 0

                                    ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)

                                    if len(plate_ocr_5_char_dict) > 0:
                                        ocr_all_number = ocr_all_number[0:-5] + str(max(plate_ocr_5_char_dict,
                                                key=plate_ocr_5_char_dict.get)) + ocr_all_number[-4:]

                                    ocr_all_number = mod_area_name(ocr_all_number)
                                    ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)
                                    ocr_front_chars = ocr_all_number[:-4]  # 뒤의 문자 4자리를 제외한 앞의 문자열

                                    if len(ocr_list) == 0:
                                        ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
                                        if log_save_s:
                                            log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)
                                        if lp_img_save:
                                            save_detect_img(crop_plate, ocr_all_number)
                                        truck_count += 1

                                        # post connect
                                        post_send_p(ocr_last_4_word)
                                        detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                        print("detect_time : ", detect_time, "OCR_last_4 : ", ocr_last_4_word,
                                              "OCR_all_number : ", ocr_all_number, "in_truck_count :", truck_count)
                                    else:
                                        score = lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, plate_id,
                                                            ocr_all_number, lp_conf)
                                        # lp_conf 값이 지정 값을 넘으면 동일 번호판 으로 인식
                                        if score >= lp_conf:
                                            pass
                                        else:
                                            ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word, plate_id]
                                            if log_save_s:  # True 일 경우  로그 저장
                                                log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)

                                            if lp_img_save:
                                                save_detect_img(crop_plate, ocr_all_number)

                                            truck_count += 1

                                            post_send_p(ocr_last_4_word)
                                            detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                            print("detect_time : ", detect_time, "OCR_last_4 : ", ocr_last_4_word,
                                                  "OCR_all_number : ", ocr_all_number, "in_truck_count :", truck_count)

                                    plate_ocr_4word_dict.clear()
                                    plate_ocr_all_number_dict.clear()
                                    plate_ocr_5_char_dict.clear()
                                    plate_detection_tracking_flag = False
                                    detection_frame_count = 0
                                    count_24 = 0
                                else:
                                    count_24 += 1
                        else:
                            plate_detection_tracking_flag = False
                            detection_frame_count = 0
                            count_24 = 0
                            plate_ocr_all_number_dict.clear()
                            plate_ocr_4word_dict.clear()
                            plate_ocr_5_char_dict.clear()
                else:
                    plate_detection_tracking_flag = False
                    detection_frame_count = 0
                    count_24 = 0
                    plate_ocr_all_number_dict.clear()
                    plate_ocr_4word_dict.clear()
                    plate_ocr_5_char_dict.clear()
            fps_str, prev_time = fps_calculator(rtsp_cur_time, prev_time)
        else:
            break

def post_send_p(ocr_result_4):
    cameraId_list = [111111111, 222222222, 333333333]
    cameraGuid_list = ['5c2de1a4-ef27-4780-8759-9ee9830b83f2',
                       'FF80F014-EDFC-499B-BDD5-A1997909DD3B',
                       '173BC97B-1780-41FC-9671-E43C2ED7C158']
    # scrapAreaCd_list = ['2100E1', '1200A1', '1200B2'] #하차구역
    scrapAreaCd_list = ['2100E1']  # 하차구역
    post_dest_url_list = ["http://ims.yksteel.co.kr:90/WebServer/AiResult",
                          "http://192.168.70.43:8080/ai/receiveVehicleInfo",
                          "http://192.168.20.216:8080/ai/receiveVehicleInfo",
                          "http://1.209.33.170:8099/get",
                          "http://imsns.idaehan.com:8080/ai/receiveVehicleInfo",
                          "https://imsns.idaehan.com:8443/ai/receiveVehicleInfo"]
    truck_schedule_url_list = ["http://ims.yksteel.co.kr:90/WebServer/MstrWait",
                               "http://192.168.70.30:8080/tms/searchIncomongVehicleList.do",
                               "https://wss.idaehan.com:8443/tms/searchIncomongVehicleList.do"]

    if ocr_result_4 is None:
        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        new_result_dict = {"procYn": "", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "",
                           "vhclNo": "", "cameraGuid": "", "scrapAreaCd": ""}
        post_send(new_result_dict, post_dest_url_list[5], scrapAreaCd_list, cameraGuid_list)
    else:
        local_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "scaleNumb": "", "carNumb": "",
                       "date": ""}
        new_result_dict = {"procYn": "N", "rcnCarNumb": "", "rcnDt": local_time, "tmsOdrNo": "", "vhclNo": ""}

        result_dict["rcnCarNumb"] = ocr_result_4
        new_result_dict["rcnCarNumb"] = ocr_result_4
        scheduled_plate = find_truck_schedule(ocr_result_4, truck_schedule_url_list[2])

        if scheduled_plate == "":
            new_result_dict["rcnCarNumb"] = ocr_result_4
            new_result_dict["procYn"] = 'N'
            post_send(new_result_dict, post_dest_url_list[5], scrapAreaCd_list, cameraGuid_list)
        else:
            print("START")
            new_result_dict["rcnCarNumb"] = ocr_result_4
            new_result_dict["procYn"] = 'Y'
            new_result_dict["vhclNo"] = scheduled_plate["vhclNo"]
            new_result_dict["tmsOdrNo"] = scheduled_plate["tmsOdrNo"]  # tmsOdrNo
            # new_post_send(new_result_dict, post_dest_url_list[1])
            # new_post_send(new_result_dict, camera_id, post_dest_url_list[2])
            post_send(new_result_dict, post_dest_url_list[5], scrapAreaCd_list, cameraGuid_list)

def post_send(data, dest_url, scrapAreaCd_list, cameraGuid_list):
    try:
        data["cameraGuid"] = cameraGuid_list[0]
        data["scrapAreaCd"] = scrapAreaCd_list[0]
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

def init_variable(plate_detection_tracking_flag, detection_frame_count, count_24, plate_ocr_all_number_dict,
                  plate_ocr_4word_dict, plate_ocr_5_char_dict):
    plate_ocr_4word_dict.clear()
    plate_ocr_all_number_dict.clear()
    plate_ocr_5_char_dict.clear()
    plate_detection_tracking_flag = False
    detection_frame_count = 0
    count_24 = 0

def lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, plate_id, ocr_all_number, lp_conf):
    score = 100  # 초기 점수를 100으로 설정
    # ocr 문자열 이 모두 동일 하면 같은 번호판 으로 판단
    if ocr_all_number == ocr_list[0]:
        score = 100  # ocr_all_number와 ocr_list[0]이 동일한 경우, 점수를 100으로 설정
        return score
    else:
        if ocr_last_4_word != ocr_list[2]:  # 뒤의 4자리가 다른 경우 차감
            for i in range(4):
                if ocr_list[2][i] != ocr_last_4_word[i]:
                    score -= 15

        # 앞의 문자 열이 다른 경우
        if ocr_list[1] != ocr_front_chars:
            # (1) 이전 문자열 길이 가 현재 문자열 보다 같거나 긺
            if len(ocr_list[1]) <= len(ocr_front_chars):
                for i in range(len(ocr_list[1])):
                    o_a_m_slice = ocr_front_chars[:len(ocr_list[1])]
                    if ocr_list[1][i] != o_a_m_slice[i]:
                        score -= 30 // len(o_a_m_slice)
                        if score < lp_conf:
                            return score

            # (2) 현재 문자열 이 이전 문자열 보다 긺
            elif len(ocr_list[1]) > len(ocr_front_chars):
                for i in range(len(ocr_front_chars)):
                    o_l_slice = ocr_list[1][:len(ocr_front_chars)]
                    if o_l_slice[i] != ocr_front_chars[i]:
                        score -= 30 // len(o_l_slice)
                    if score < lp_conf:
                        return score

        return score

        # id 값이 다른 경우
        # score -= 10 if plate_id != ocr_list[3] else 0

    # return score


def mod_area_name(ocr_all_number):
    # 수정할 대상 문자 및 대체할 문자를 딕셔너리로 정의
    error_correction_dict_1 = {'누': '부', '무': '부', '천': '전', '수': '부', '주': '충'}
    error_correction_dict_2 = {'수': '주', '가': '기', '님': '남', '묵': '북', '사': '산'}

    # ocr_all_number의 첫 번째 문자가 수정 대상 문자인 경우, 해당 문자를 대체 문자로 변경
    if ocr_all_number[0] in error_correction_dict_1.keys():
        corrected_char = error_correction_dict_1[ocr_all_number[0]]
        ocr_all_number = corrected_char + ocr_all_number[1:]
        return ocr_all_number

    # ocr_all_number의 두 번째 문자가 수정 대상 문자인 경우, 해당 문자를 대체 문자로 변경
    elif ocr_all_number[1] in error_correction_dict_2.keys():
        corrected_char = error_correction_dict_2[ocr_all_number[1]]
        ocr_all_number = ocr_all_number[0] + corrected_char + ocr_all_number[2:]
        return ocr_all_number

    else:
        return ocr_all_number


def log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict):
    with open('ocr_log.txt', 'a') as log_f:
        time_stream = time.strftime('%Y-%m-%d %H:%M:%S')
        log_f.write("[" + time_stream + "]" + " OCR_LAST_4 : " + str(
            ocr_last_4_word) +
                    "\n")
        if len(plate_ocr_5_char_dict) > 0:
            ocr_all_number = ocr_all_number[0:-5] + str(
                max(plate_ocr_5_char_dict,
                    key=plate_ocr_5_char_dict.get)) + ocr_all_number[-4:]
        log_f.write(
            "[" + time_stream + "]" + " OCR_ALL_NUMBER : " + str(
                ocr_all_number) +
            "\n")


def save_detect_img(crop_plate, ocr_all_number):
    # 현재 디렉토리 경로
    root_dir = os.getcwd()

    # detect_plate 디렉토리 경로
    detect_plate_dir = os.path.join(root_dir, 'detect_plate')

    # detect_plate 디렉토리가 없다면 생성
    if not os.path.exists(detect_plate_dir):
        os.makedirs(detect_plate_dir)

    # 오늘 날짜(yyyy-mm-dd)로 폴더 이름 생성
    today = datetime.date.today().strftime('%Y-%m-%d')
    today_dir = os.path.join(detect_plate_dir, today)

    # 오늘 날짜로 된 폴더가 없다면 생성
    if not os.path.exists(today_dir):
        os.mkdir(today_dir)

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    crop_plate = Image.fromarray(crop_plate.astype(np.uint8))
    crop_plate.save(os.path.join(today_dir, f'{time_str + "_" + ocr_all_number}.jpg'), quality=1)


def check_path(camera_path, log_save_s, lp_conf, lp_img_save):
    if camera_path:
        input_rtsp(camera_path, log_save_s, lp_conf, lp_img_save)
    else:
        print("Enter the camera path!")


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default=None, help='set your camera IP')
    parser.add_argument('--log_save', default=False, help='You can choose log save.')
    parser.add_argument('--conf', type=float, default=70, help='Same license plate accuracy out of 100.')
    parser.add_argument('--img_save', default=False, help='Save the crop image of the detected license plate displayed '
                                                          'in the log')
    args = parser.parse_args()
    return args


def record_video(camera_path):
    rtsp = cv2.VideoCapture(camera_path)
    while True:
        rtsp_ret, rtsp_img = rtsp.read()

        if rtsp_ret:
            resize_original_img = cv2.resize(rtsp_img, (640, 480))
            cv2.imshow("record", resize_original_img)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
            break


def main():
    args = input_parser()
    camera_path = args.camera
    log_save_s = args.log_save
    lp_conf = args.conf
    lp_img_save = args.img_save
    # camera_path = "Z:\\sian\\accuracy_video_cap\\orange_yellow1.mp4"
    check_path(camera_path, log_save_s, lp_conf, lp_img_save)
    ### 사용안함 <오류수정중>
    # multi_rtsp = Multi_processing_rtsp_cls()
    # multi_rtsp.multi_processing_rtsp()


if __name__ == '__main__':
    main()
