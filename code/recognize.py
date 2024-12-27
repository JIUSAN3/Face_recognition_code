import os
import cv2
import numpy as np
import dlib
import mss
from keras.models import load_model
import pygetwindow as gw

# 加载训练好的模型
model = load_model('face_recognition_model.h5')

# 初始化人脸检测器
detector = dlib.get_frontal_face_detector()

# 定义图像大小
image_size = (64, 64)

def preprocess_image(img):
    """
    预处理图像：灰度化、调整大小、归一化。

    :param img: 输入图像
    :return: 预处理后的图像
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, image_size)
    normalized_img = resized_img / 255.0
    return normalized_img.reshape(-1, image_size[0], image_size[1], 1)

def recognize_faces(img):
    """
    识别图像中的人脸。

    :param img: 输入图像
    :return: 识别结果和置信度
    """
    # 检测人脸
    dets = detector(img, 1)

    for i, d in enumerate(dets):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()

        # 裁剪人脸区域
        face = img[y1:y2, x1:x2]

        # 预处理人脸图像
        preprocessed_face = preprocess_image(face)

        # 进行人脸识别
        predictions = model.predict(preprocessed_face)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]

        # 标记人脸区域和识别结果
        label = f'Class {class_index}: {confidence:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return img

def main():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 选择第一个显示器

        while True:
            # 获取当前活动窗口的标题
            active_window_title = gw.getActiveWindowTitle()

            # 如果当前活动窗口是 "Face Recognition"，则跳过截图
            if active_window_title == "Face Recognition":
                continue

            # 获取屏幕截图
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)

            # 转换为OpenCV格式
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # 识别人脸
            result_img = recognize_faces(img)

            # 显示结果
            cv2.imshow('Face Recognition', result_img)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
