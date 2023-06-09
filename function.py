# -*- coding:utf-8 -*-
import itertools
import random
from copy import copy
from random import randint

import cv2
import numpy as np
from PIL import Image


def check_class_bright_control(crop_plate, c):
    cp_gray = cv2.cvtColor(crop_plate, cv2.COLOR_BGR2GRAY)  # 회색 조로 변경 하고 밝기 측정
    # 밝기 조절
    brightness = int(np.mean(cp_gray))
    permit_bright = 130  # 밝기의 범위는 0 ~ 255임 평균인 130을 기준 으로 정해둠

    r = permit_bright - brightness
    if r > 0:  # 양수
        cp = cv2.add(cp_gray, r)
    else:  # 음수 ,0
        cp = cv2.subtract(cp_gray, abs(r))

    #  대비
    alpha = 3.0
    control_cp = np.clip((1 + alpha) * cp - 128 * alpha, 0, 255).astype(np.uint8)

    if c == 0 or c == 1 or c == 5 or c == 6:  # 초록,주황 번호판(하얀 문자)
        cp = ~control_cp
        return cp
    else:
        return control_cp


def threshold_blur(cp):
    img_blurred = cv2.GaussianBlur(cp, ksize=(5, 5), sigmaX=0)

    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=19,
        C=9
    )
    return img_blur_thresh


def find_contour(img_blur_thresh):  # 컨투어 찾기

    contours, _ = cv2.findContours(
        img_blur_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    contours_dict = []  # 컨투어와 컨투어를 감싸는 사각형에 대한 정보를 저장
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        contours_dict.append({  # 리스트안에 딕셔너리 추가
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    return contours_dict


def find_lp_chars(cp, cpw, cph):  # 번호판 문자 찾기
    img_blur_thresh = threshold_blur(cp)
    contours_dict = find_contour(img_blur_thresh)
    possible_contours = []
    cnt = 0
    for c_d in contours_dict:
        area = c_d['w'] * c_d['h']  # 넓이
        ratio = c_d['w'] / c_d['h']  # 비율
        plate_area = cpw * cph
        #  컨투어 박스 면적 범위 조정
        if 1 < area < plate_area // 8 and 0.3 < ratio < 0.9:
            c_d['idx'] = cnt
            cnt += 1
            possible_contours.append(c_d)

    matched_result = []  # 컨투어 박스 안에 컨투어 박스가 있는건 제외 하고 담기
    for p_c in range(len(possible_contours)):
        inside_box = False
        for j in range(len(possible_contours)):
            if p_c != j and (possible_contours[p_c]['x'] > possible_contours[j]['x']) and (
                    possible_contours[p_c]['y'] > possible_contours[j]['y']) and (
                    possible_contours[p_c]['w'] < possible_contours[j]['w']) and (
                    possible_contours[p_c]['h'] < possible_contours[j]['h']) and (
                    possible_contours[p_c]['y'] + possible_contours[p_c]['h'] <
                    possible_contours[j]['y'] + possible_contours[j]['h']
            ):
                inside_box = True
                break
        if not inside_box:
            matched_result.append(possible_contours[p_c])

    #  가장 긴 컨투어 박스 세로 길이의 0.95배 이상인 컨투어 박스만 담기
    highest_char_h = sorted(matched_result, key=lambda x: x['h'])
    matched_chars = []
    for mc in matched_result:
        if mc['h'] >= (highest_char_h[-1]['h'] * 0.95) and mc['y'] > 0:
            matched_chars.append(mc)

    return matched_chars


def distortion_correction(mask, im0):
    mask_np = mask.cpu().numpy()[0]  # GPU 상의 Tensor를 CPU 상의 NumPy 배열로 변환
    points_list = cv2.findNonZero((mask_np * 255).astype(np.uint8)).tolist()  # 좌표 리스트 반환
    points_list = [coord[0] for coord in points_list]

    sum_point_list = []
    diff_point_list = []
    for points in points_list:
        sum_pont = sum(points)
        diff_point = points[1] - points[0]
        sum_point_list.append(sum_pont)
        diff_point_list.append(diff_point)

    max_sum_index = sum_point_list.index(max(sum_point_list))
    min_sum_index = sum_point_list.index(min(sum_point_list))
    max_diff_index = diff_point_list.index(max(diff_point_list))
    min_diff_index = diff_point_list.index(min(diff_point_list))

    # x+y 최소 값이 좌하단//좌상단
    topLeft = points_list[min_sum_index]

    # x+y 최대 값이 우상단 좌표 // 우하단
    bottomRight = points_list[max_sum_index]

    # y-x 최소 값이 우하단 좌표  //우상단
    topRight = points_list[min_diff_index]

    # y-x 최대 값이 좌상단 좌표 //좌하단
    bottomLeft = points_list[max_diff_index]

    # 변환 전 번호판 4개 좌표
    origin_plate_point = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 번호판의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])  # 두 좌우 거리 간의 최대 값이 번호판 의 폭
    height = max([h1, h2])  # 두 상하 거리 간의 최대 값이 번호판 의 높이

    # 변환 후 번호판 4개 좌표
    change_plate_point = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # 변환 행렬 계산
    mtrx = cv2.getPerspectiveTransform(origin_plate_point, change_plate_point)

    # 원근 변환 적용
    img_distorted = cv2.warpPerspective(im0, mtrx, (width, height))

    return img_distorted


def rotation_correction(matched_result, cp, cpw, cph):
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])

    top_right_x_list = []  # 우상단 x좌표 값을 담을 리스트
    for x_c in sorted_x_chars:
        top_right_x_list.append(x_c['x'] + x_c['w'])  # 우상단 좌표 x값을 리스트에 담기
    index_rightmost = top_right_x_list.index(max(top_right_x_list))  # 가장 우측의 x값을 가진 인덱스

    plate_cx = (sorted_x_chars[0]['cx'] + sorted_x_chars[index_rightmost]['cx']) // 2  # 번호판 중앙 x 좌표
    plate_cy = (sorted_x_chars[0]['cy'] + sorted_x_chars[index_rightmost]['cy']) // 2  # 번호판 중심 y 좌표

    # 번호 판의 기율어진 각도 구하기
    triangle_height = sorted_x_chars[index_rightmost]['cy'] - sorted_x_chars[0]['cy']  # 삼각함수 사용
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_x_chars[0]['cx'], sorted_x_chars[0]['cy']]) -
        np.array([sorted_x_chars[index_rightmost]['cx'], sorted_x_chars[index_rightmost]['cy']])
    )

    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle,
                                              scale=1.0)
    img_rotated = cv2.warpAffine(cp, M=rotation_matrix,
                                 dsize=(cpw, cph))

    return img_rotated


def classes_division(matched_result, img_distorted, cpw, cph, c):  # 번호판 종류 구분
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])
    sorted_y_chars = sorted(matched_result, key=lambda y: y['y'])
    # contour box 개수가 3개 이상 일때 문자열 사이 크기 구하기
    if len(matched_result) >= 3:
        # ko_size = int(sorted_x_chars[2]['x'] - sorted_x_chars[1]['x'] + sorted_x_chars[1]['w'])
        ko_size = (int(sorted_x_chars[2]['x'] - sorted_x_chars[1]['x'] + sorted_x_chars[1]['w']) *
                   int(sorted_y_chars[-1]['y'] + sorted_y_chars[-1]['h'] - sorted_y_chars[0]['y']))
    # contour box 가 4개 이상 이면 기울기 보정
    if len(matched_result) >= 4:
        img_distorted = rotation_correction(matched_result, img_distorted, cpw, cph)

    if c == 3 or c == 5 or c == 6:  # (2줄 번호판)
        top_crop_img, under_crop_img = crop_composite(img_distorted, c, matched_result)
        attach_width_crop_img = cv2.hconcat([top_crop_img, under_crop_img])

        return attach_width_crop_img

    elif c == 2:  # 노랑 1줄
        # 세로 쓰기 한글 전 처리
        vertical_composite_img = vertical_orientation_composite(matched_result, img_distorted)

        # contour box 가 최소 3개 이상 이고 2 번째 와 3번째 contour box 사이가 존재 해야 함
        if len(matched_result) >= 3 and ko_size > 0:
            attach_stuck_ko_img = find_stuck_ko(matched_result, img_distorted)
            vertical_composite_img = cv2.resize(vertical_composite_img, (
                int(vertical_composite_img.shape[1] * vertical_composite_img.shape[0] // attach_stuck_ko_img.shape[0]),
                attach_stuck_ko_img.shape[0]))
            attach_img = cv2.hconcat([vertical_composite_img, attach_stuck_ko_img])
            return attach_img

        else:  # 문자열 사이에 낀 한글 보정 불가 시
            x = sorted_x_chars[0]['x']  # 제일 앞에 x 값
            img_cropped = img_distorted[0: img_distorted.shape[0], x: img_distorted.shape[1]]
            vertical_composite_img = cv2.resize(vertical_composite_img, (
                int(vertical_composite_img.shape[1] * vertical_composite_img.shape[0] // img_cropped.shape[0]),
                img_cropped.shape[0]))
            attach_img = cv2.hconcat([vertical_composite_img, img_cropped])
            return attach_img

    elif c == 4:  # 하양 1줄
        if len(matched_result) >= 3 and ko_size > 0:
            attach_stuck_ko_img = find_stuck_ko(matched_result, img_distorted)
            return attach_stuck_ko_img
        else:
            return img_distorted

    else:  # 주황, 초록 new
        top_crop_img, under_crop_img = crop_composite(img_distorted, c, matched_result)

        if len(matched_result) >= 3 and ko_size > 0:
            attach_stuck_ko_img = find_stuck_ko(matched_result, img_distorted)
            top_crop_img = cv2.resize(top_crop_img, (
                int(top_crop_img.shape[1] * top_crop_img.shape[0] // attach_stuck_ko_img.shape[0]),
                attach_stuck_ko_img.shape[0]))

            attach_width_crop_img = cv2.hconcat([top_crop_img, attach_stuck_ko_img])
            return attach_width_crop_img

        else:
            attach_width_crop_img = cv2.hconcat([top_crop_img, under_crop_img])
            return attach_width_crop_img


def crop_composite(img_rotated, c, matched_result):  # 2줄 번호판 윗줄 아랫 줄 crop , 합성
    irh, irw = img_rotated.shape
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])
    y = sorted_x_chars[0]['y']  # 맨 앞 컨투어 의 y 값

    if c == 3 or c == 6:  # yellow 2line 과 green top_word는 윗줄 + 아랫줄 작업만 진행.
        if len(matched_result) >= 4:
            top_crop_img = img_rotated[0: y, int(sorted_x_chars[0]['x'] * 0.9): sorted_x_chars[3]['x']]
        else:
            top_crop_img = img_rotated[0: y, int(irw * 0.2): int(irw * 0.8)]  # 윗문자 crop
        under_crop_img = img_rotated[y: int(irh), int(irw * 0.02): int(irw * 0.98)]

        top_crop_h, top_crop_w = top_crop_img.shape
        under_crop_h, under_crop_w = under_crop_img.shape  # 아래 crop한 부분의 세로 가로 값
        # 아래 문자열 의 가로 비율 조정
        under_crop_mod_img = cv2.resize(under_crop_img, (int(under_crop_w * 1.6), under_crop_h))

        under_crop_mod_h, under_crop_mod_w = under_crop_mod_img.shape
        top_crop_img = cv2.resize(top_crop_img, (int(top_crop_w * under_crop_mod_h / top_crop_h), under_crop_mod_h))

        return top_crop_img, under_crop_mod_img
    else:
        if len(matched_result) >= 5:
            top_crop_left = img_rotated[0: y, sorted_x_chars[1]['x']: sorted_x_chars[2]['x']]
            top_crop_right = img_rotated[0: y, sorted_x_chars[3]['x']: sorted_x_chars[4]['x'] + sorted_x_chars[4]['w']]
        else:
            top_crop_left = img_rotated[0: y, int(irw * 0.2): int(irw * 0.45)]
            top_crop_right = img_rotated[0: y, int(irw * 0.6): int(irw * 0.8)]
        under_crop_img = img_rotated[irh // 2: int(irh), int(irw * 0.02): int(irw * 0.98)]
        under_crop_h, under_crop_w = under_crop_img.shape

        composite_crop_img = cv2.hconcat([top_crop_left, top_crop_right])  # 윗글자 왼쪽 오른쪽 합성
        composite_h, composite_w = composite_crop_img.shape

        composite_crop_img = cv2.resize(composite_crop_img, (composite_w * under_crop_h // composite_h, under_crop_h))

        return composite_crop_img, under_crop_img


def vertical_orientation_composite(matched_result, img_rotated):  # 세로 쓰기 되어 있는 한글 crop, 합성
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])
    x = sorted_x_chars[0]['x']  # 제일 앞에 x 값

    irh, irw = img_rotated.shape
    if len(matched_result) > 3:
        # 세로 쓰기 한글 영역
        vertical_crop_img = img_rotated[int(irh * 0.1): int(irh - (irh * 0.1)), int(x * 0.2): x]
    else:
        vertical_crop_img = img_rotated[int(irh * 0.1): int(irh - (irh * 0.1)), irw // 30: irw // 6]
    vertical_crop_h, vertical_crop_w = vertical_crop_img.shape

    # 세로 쓰기 한글 윗 부분
    first_ko = vertical_crop_img[0: vertical_crop_h // 2, 0: vertical_crop_w]
    first_ko_h, first_ko_w = first_ko.shape

    # 세로 쓰기 한글 아랫 부분
    last_ko = vertical_crop_img[int(first_ko_h): int(vertical_crop_h), 0: vertical_crop_w]

    # 위 -> 앞 / 아래 -> 뒤
    front_img = cv2.resize(first_ko, (vertical_crop_w * 3, vertical_crop_h // 2))
    behind_img = cv2.resize(last_ko, (vertical_crop_w * 3, vertical_crop_h // 2))

    # 앞 글자 뒷 글자 합성
    vertical_composite_img = cv2.hconcat([front_img, behind_img])

    return vertical_composite_img


def find_stuck_ko(matched_result, img_distorted):  # 번호판 사이 한글이 끼여 있는 경우(컨투어 박스 활용)
    sorted_x_chars = sorted(matched_result, key=lambda x: x['x'])
    sorted_y_chars = sorted(matched_result, key=lambda y: y['y'])

    # 사이에 낀 한글 영역
    stuck_ko_img = img_distorted[sorted_y_chars[0]['y']: sorted_y_chars[-1]['y'] + sorted_y_chars[-1]['h'],
                   sorted_x_chars[1]['x'] + sorted_x_chars[1]['w']:sorted_x_chars[2]['x']]
    ko_size = sorted_x_chars[2]['x'] - sorted_x_chars[1]['x'] + sorted_x_chars[1]['w']
    stuck_ko_img = cv2.resize(stuck_ko_img, (int(ko_size * 1.2), stuck_ko_img.shape[0]))
    # 앞 문자열
    front_crop = img_distorted[sorted_y_chars[0]['y']: sorted_y_chars[-1]['y'] + sorted_y_chars[-1]['h'],
                 sorted_x_chars[0]['x']:sorted_x_chars[1]['x'] + sorted_x_chars[1]['w']]
    # 뒷 문자열
    back_crop = img_distorted[sorted_y_chars[0]['y']: sorted_y_chars[-1]['y'] + sorted_y_chars[-1]['h'],
                sorted_x_chars[2]['x']:int(img_distorted.shape[1] * 0.98)]

    front_crop = cv2.resize(front_crop, (int(front_crop.shape[1] * 1.2), front_crop.shape[0]))  # 1.2배
    back_crop = cv2.resize(back_crop, (int(back_crop.shape[1] * 1.4), back_crop.shape[0]))  # 1.4배
    # 앞 문자열 + 사이에 낀 한글 + 뒷 문자열
    attach_stuck_ko_img = cv2.hconcat([front_crop, stuck_ko_img, back_crop])

    return attach_stuck_ko_img

def draw_contour_box(img_distorted, matched_chars):  # contour box 확인용 함수
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    rgb_image = cv2.cvtColor(img_distorted, cv2.COLOR_GRAY2RGB)

    for m_c in matched_chars:
        rgb_image = cv2.rectangle(rgb_image, (m_c['x'], m_c['y']), (m_c['x'] + m_c['w'], m_c['y'] + m_c['h']),
                                  (red, green, blue), 3)

    save_part = Image.fromarray(rgb_image.astype(np.uint8))
    save_part.save('C:/yolo_ocr-master_test/yolov7/output_ocr_pre_f/rgb_image.jpg',
                   "JPEG")
