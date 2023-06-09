# -*- coding:utf-8 -*-
import cv2
import time
import lp_detection_tracking
import numpy as np
import string
import argparse

def video_show(original_img, infer_img, fps_str, window_name):
    # resize video
    resize_original_img = cv2.resize(original_img, (640, 480))
    resize_infer_img = cv2.resize(infer_img, (640, 480))

    # Check FPS
    cv2.putText(resize_original_img, fps_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Convert to one Window
    numpy_horizontal_img = np.hstack((resize_original_img, resize_infer_img))
    cv2.imshow(window_name, numpy_horizontal_img)
    cv2.waitKey(1)


def fps_calculator(cur_time, prev_time):
    sec = cur_time - prev_time
    prev_time = cur_time

    fps = 1 / (sec)
    fps_str = "FPS : %0.1f" % fps
    return fps_str, prev_time


def input_rtsp(camera_path,log_save_s):
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
    del_id_list = []
    detect_plate_id_dict = {}
    plate_ocr_4word_dict = {}
    plate_ocr_all_number_dict = {}
    plate_ocr_5_char_dict = {}
    ocr_list = []
    while True:
        rtsp_ret, rtsp_img = rtsp.read()
        rtsp_cur_time = time.time()

        if rtsp_ret:
            """
            딕셔너리를 확인하고 설정된 프레임을 만족한 키 값을 삭제하기 위함
            딕셔너리를 탐지하고 있는동안에는 바로 삭제가 불가능, 리스트를 만들어 아이디 값을 저장하고 저장된 아이디를 이용 딕셔너리 삭제
            """
            if detect_plate_id_dict is not None:
                for key, value in detect_plate_id_dict.items():
                    if value[1] == True:
                        del_id_list.append(key)

            if len(del_id_list) > 0:
                for del_id_key in del_id_list:
                    if del_id_key in detect_plate_id_dict:
                        del detect_plate_id_dict[del_id_key]

            if detection_frame_count == 30:
                plate_detection_tracking_flag = True

            # plate_id_list = [id, ocr_words]
            rtsp_infer_img, plate_detection_flag, plate_id_list = \
                detection_yolo.inference_img(rtsp_img, plate_detection_tracking_flag=plate_detection_tracking_flag)

            if not plate_detection_flag:
                detection_frame_count = 0

            if plate_detection_flag and not plate_detection_tracking_flag:
                detection_frame_count += 1

            if plate_detection_tracking_flag:
                for index, value in enumerate(plate_id_list):
                    plate_id = value[0]
                    ocr_word = value[1]

                    plate_id = 'plate_id_' + str(plate_id)
                    # print("plate_id : ", plate_id)
                    # print("continue : ", del_id_list)

                    if plate_id in del_id_list:
                        plate_detection_tracking_flag = False
                        detection_frame_count = 0
                        continue
                    else:
                        del_id_list.clear()

                        if len(ocr_word) > 5:
                            all_plate_number = ocr_word.translate(str.maketrans('', '', string.punctuation))
                            all_plate_number = all_plate_number.replace(" ", "")
                            last_4_word = all_plate_number[-4:]

                            # last 4 word score
                            if last_4_word not in plate_ocr_4word_dict:
                                plate_ocr_4word_dict[last_4_word] = 1
                            else:
                                plate_ocr_4word_dict[last_4_word] += 1

                            # all number score, 5th korean check
                            if all_plate_number not in plate_ocr_all_number_dict:
                                if all_plate_number[-5].isalpha():
                                    plate_ocr_5_char_dict[all_plate_number[-5]] = 1
                                plate_ocr_all_number_dict[all_plate_number] = 1
                            else:
                                if all_plate_number[-5].isalpha():
                                    plate_ocr_5_char_dict[all_plate_number[-5]] += 1
                                plate_ocr_all_number_dict[all_plate_number] += 1

                            if plate_id not in detect_plate_id_dict:
                                detect_plate_id_dict[plate_id] = [1, False]

                            else:
                                if detect_plate_id_dict[plate_id][0] == 24:
                                    detect_plate_id_dict[plate_id][1] = True
                                    ocr_all_number = max(plate_ocr_all_number_dict, key=plate_ocr_all_number_dict.get)
                                    ocr_last_4_word = max(plate_ocr_4word_dict, key=plate_ocr_4word_dict.get)

                                    if len(ocr_list) == 0:
                                        ocr_list = [ocr_all_number]
                                        detect_plate_id_dict[plate_id][0] += 1

                                        if log_save_s:  # True 일 경우  로그 저장
                                            log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)

                                        print("OCR_last_4 : ", ocr_last_4_word)
                                        print("OCR_all_number : ", ocr_all_number)

                                    else:  # 리스트가 비어 있지 않은 경우
                                        ocr_chars = ''.join(ocr_list)  # list 를 string 으로 변환

                                        # 이전에 인식된 ocr 문자와 현재 인식된 ocr 문자가 같은지 확인
                                        if ocr_all_number == ocr_chars:
                                            continue

                                        else:  # 같지 않다면 아래 진행
                                            ocr_list = [ocr_all_number]  # 현재 인식된 문자를 ocr_list 로 재정의 하여 저장

                                            if log_save_s:  # True 일 경우  로그 저장
                                                log_save(ocr_last_4_word, ocr_all_number, plate_ocr_5_char_dict)

                                            print("OCR_last_4 : ", ocr_last_4_word)
                                            print("OCR_all_number : ", ocr_all_number)

                                    plate_ocr_4word_dict.clear()
                                    plate_ocr_all_number_dict.clear()
                                    plate_ocr_5_char_dict.clear()
                                    plate_detection_tracking_flag = False
                                    detection_frame_count = 0

                                else:
                                    detect_plate_id_dict[plate_id][0] += 1

            fps_str, prev_time = fps_calculator(rtsp_cur_time, prev_time)
        else:
            break


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


def check_path(camera_path, log_save_s):
    if camera_path:
        input_rtsp(camera_path, log_save_s)
    else:
        print("Enter the camera path!")


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera', default='test/test.mp4', help='set your camera IP')
    parser.add_argument('--log_save', default=False, help='You can choose log save.')
    args = parser.parse_args()
    return args


def main():
    args = input_parser()
    camera_path = args.camera
    log_save_s = args.log_save
    check_path(camera_path, log_save_s)


if __name__ == '__main__':
    # app.run(debug=True)
    main()
