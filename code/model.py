import os
import cv2
import dlib
import numpy as np

def load_and_preprocess_dataset(data_dir, output_dir, image_size=(64, 64)):
    """
    加载数据集并进行预处理，将检测到的面孔截取出来存储在指定目录中。

    :param data_dir: 数据集目录
    :param output_dir: 输出目录，用于存储截取的面孔图像
    :param image_size: 图像大小，默认为 (64, 64)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = os.listdir(data_dir)
    detector = dlib.get_frontal_face_detector()
    face_count = 0

    for category in categories:
        path = os.path.join(data_dir, category)

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # 读取图像并转换为灰度图像
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, image_size)

                # 使用dlib进行人脸检测，dets为返回的结果
                dets = detector(img_array, 1)

                # 使用enumerate 函数遍历序列中的元素以及它们的下标，下标i即为人脸序号
                for i, face in enumerate(dets):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_img = img_array[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, image_size)

                    # 保存截取的面孔图像
                    output_face_path = os.path.join(output_dir, f'face_{face_count}.jpg')
                    cv2.imwrite(output_face_path, face_img)
                    face_count += 1

            except Exception as e:
                print(f"Error processing image {img_name}: {e}")

if __name__ == "__main__":
    data_dir = r'D:\face recognize project\lfw'
    output_dir = 'output_faces'
    load_and_preprocess_dataset(data_dir, output_dir)
