# -*- coding:utf-8 -*-
import datetime
import os
import cv2
import time
from PIL import Image
# import lp_detection_tracking
import lp_detection_model
import numpy as np
import string
import argparse
import logging
from kafka import KafkaProducer
import traceback
from json import dumps

logging.basicConfig(level=logging.INFO)

def fps_calculator(cur_time, prev_time):
    sec = cur_time - prev_time
    prev_time = cur_time

    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    return fps_str, prev_time


def input_rtsp(camera_path, lp_conf, bs, topic):
    prev_time = 0
    detection_yolo = lp_detection_model.lp_detection_tracking()
    detection_yolo.model_init()
    rtsp = cv2.VideoCapture(camera_path)

    # rtsp://admin:admin13579!@220.95.111.220/profile2/media.smp
    camera_path_split = camera_path.split("@")
    camera_id = ""
    if camera_path_split[1] == "220.95.111.220/profile2/media.smp":
        camera_id = "1"
    elif camera_path_split[1] == "192.168.0.15":
        camera_id = "2"
    elif camera_path_split[1] == "192.168.0.13":
        camera_id = "3"

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
    # between_ko = None
    ko_list = '가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주바사아자허하호배'
    post_frame_count = 0

    # Kafka producer
    producer = KafkaProducer(acks=0, bootstrap_servers=[bs], api_version=(0, 11, 5),
                             value_serializer=lambda x: dumps(x).encode('utf-8'))

    while True:
        rtsp_ret, rtsp_img = rtsp.read()
        rtsp_cur_time = time.time()

        if rtsp_ret:
            if detection_frame_count == 30:
                plate_detection_tracking_flag = True

            if post_frame_count == 36:
                detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                producer_value = {"Detect_Time": detect_time,
                                  "Ocr_Result_4": "",
                                  "Ocr_Result_All": "",
                                  "Truck_Count": "",
                                  "Camera_id": camera_id
                                  }
                try:
                    print("send : ", producer_value)
                    response = producer.send(topic=topic, value=producer_value).get()
                    print("kafka_response : ", response)
                except:
                    traceback.print_exc()
                # print("detect_time : ", detect_time, ", OCR_last_4 : ", "None",
                #       ", OCR_all_number : ", ", in_truck_count :", truck_count,
                #       ", Camera_ID : ", camera_id)
                post_frame_count = 0

            rtsp_infer_img,plate_detection_flag, plate_id_list = detection_yolo.inference_img2(rtsp_img)

            if plate_detection_flag:
                detection_frame_count += 1
                post_frame_count = 0
            else:
                detection_frame_count = 0
                post_frame_count += 1
            #
            # print("detection_frame_count :", detection_frame_count, ", "
            #       "plate_detection_tracking_flag :", plate_detection_tracking_flag, ","
            #       "count_24 :", count_24, ","
            #       "plate_id_list : ", plate_id_list,
            #       "post_frame_count : ", post_frame_count,
            #       "ocr_list : ", ocr_list)

            if plate_detection_tracking_flag:
                if plate_id_list:
                    for index, value in enumerate(plate_id_list):
                        ocr_word = value
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
                                detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                producer_value = {"Detect_Time": detect_time,
                                                  "Ocr_Result_4": last_4_word,
                                                  "Ocr_Result_All": "",
                                                  "Truck_Count": truck_count,
                                                  "Camera_id": camera_id
                                                  }
                                try:
                                    print("send : ",producer_value)
                                    response = producer.send(topic=topic, value=producer_value).get()
                                    print("kafka_response : ", response)
                                except:
                                    traceback.print_exc()
                                # print("detect_time : ", detect_time, ", OCR_last_4 : ", last_4_word,
                                #       ", OCR_all_number : ", ", in_truck_count :", truck_count,
                                #       ", Camera_ID : ", camera_id)
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
                                    ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)

                                    if len(plate_ocr_5_char_dict) > 0:
                                        ocr_all_number = ocr_all_number[0:-5] + str(max(plate_ocr_5_char_dict,
                                                key=plate_ocr_5_char_dict.get)) + ocr_all_number[-4:]

                                    ocr_all_number = mod_area_name(ocr_all_number)
                                    ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)
                                    ocr_front_chars = ocr_all_number[:-4]  # 뒤의 문자 4자리를 제외한 앞의 문자열

                                    if len(ocr_list) == 0:
                                        ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word]
                                        truck_count += 1
                                        detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                        producer_value = {"Detect_Time": detect_time,
                                                          "Ocr_Result_4": ocr_last_4_word,
                                                          "Ocr_Result_All": ocr_all_number,
                                                          "Truck_Count": truck_count,
                                                          "Camera_id": camera_id
                                                          }
                                        try:
                                            print("send : ", producer_value)
                                            response = producer.send(topic=topic, value=producer_value).get()
                                            print("kafka_response : ", response)
                                        except:
                                            traceback.print_exc()
                                        # print("detect_time : ", detect_time, ", OCR_last_4 : ", ocr_last_4_word,
                                        #       ", OCR_all_number : ", ocr_all_number, ", in_truck_count :", truck_count,
                                        #       ", Camera_ID : ", camera_id)
                                    else:
                                        score = lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word,
                                                            ocr_all_number, lp_conf)
                                        # lp_conf 값이 지정 값을 넘으면 동일 번호판 으로 인식
                                        if score >= lp_conf:
                                            pass
                                        else:
                                            ocr_list = [ocr_all_number, ocr_front_chars, ocr_last_4_word]
                                            truck_count += 1

                                            detect_time = time.strftime('%Y-%m-%d %H:%M:%S')
                                            producer_value = {"Detect_Time": detect_time,
                                                              "Ocr_Result_4": ocr_last_4_word,
                                                              "Ocr_Result_All": ocr_all_number,
                                                              "Truck_Count": truck_count,
                                                              "Camera_id": camera_id
                                                              }
                                            try:
                                                print("send : ", producer_value)
                                                response = producer.send(topic=topic, value=producer_value).get()
                                                print("kafka_response : ", response)
                                            except:
                                                traceback.print_exc()
                                            # print("detect_time : ", detect_time, ", OCR_last_4 : ", ocr_last_4_word,
                                            #       ", OCR_all_number : ", ocr_all_number, ", in_truck_count :",
                                            #       truck_count,
                                            #       ", Camera_ID : ", camera_id)

                                    plate_ocr_4word_dict.clear()
                                    plate_ocr_all_number_dict.clear()
                                    plate_ocr_5_char_dict.clear()
                                    plate_detection_tracking_flag = False
                                    detection_frame_count = 0
                                    count_24 = 0
                                    between_ko = None
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

def lp_decision(ocr_front_chars, ocr_list, ocr_last_4_word, ocr_all_number, lp_conf):
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


def check_path(camera_path, lp_conf, bs, topic):
    if camera_path:
        input_rtsp(camera_path, lp_conf, bs, topic)
    else:
        print("Enter the camera path!")


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default=None, help='set your camera IP')
    parser.add_argument('--conf', type=float, default=70, help='Same license plate accuracy out of 100.')
    parser.add_argument('-bs', '--bootstrap_server', default=None, help="use bootstrap_server")
    parser.add_argument('-t', '--topic', default=None, help="use topic")
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
    lp_conf = args.conf
    bs = args.bootstrap_server
    topic = args.topic
    check_path(camera_path, lp_conf, bs, topic)

if __name__ == '__main__':
    main()
