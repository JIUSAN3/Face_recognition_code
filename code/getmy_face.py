import cv2
import os
import dlib
import numpy as np

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """调整图像的亮度和对比度"""
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image

def capture_and_save_images(output_dir, num_images=2000, brightness=0, contrast=0, confidence_threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        # 调整亮度和对比度
        frame = adjust_brightness_contrast(frame, brightness, contrast)

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用dlib进行人脸检测
        faces = detector(gray, 1)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (64, 64))

            # 保存图像
            image_path = os.path.join(output_dir, f'image_{count}.jpg')
            cv2.imwrite(image_path, face_img)
            count += 1

            # 显示图像
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Saved: {count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Capture Image', frame)

        # 按下'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = 'captured_images'
    capture_and_save_images(output_dir, brightness=30, contrast=30, confidence_threshold=0.5)
