import logging
from datetime import datetime
from functools import lru_cache
import yaml
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
import time
import os

# 配置日志
logging.basicConfig(
    filename=f'face_recognition_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS支持


# 加载配置文件
def load_config():
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # 如果配置文件不存在，创建默认配置
        default_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False
            },
            'model_path': 'face_recognition_model.h5',
            'categories': ['class_0', 'class_1'],
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }

        # 保存默认配置到文件
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True)

        return default_config
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        raise


# 加载配置
config = load_config()

# 加载人脸检测器
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error loading face cascade classifier")
    logging.info("Face cascade classifier loaded successfully")
except Exception as e:
    logging.error(f"Error loading face cascade classifier: {str(e)}")
    raise


# 加载模型
@lru_cache(maxsize=1)
def load_face_model():
    try:
        custom_objects = {
            'BatchNormalization': BatchNormalization,
            'Adam': Adam
        }
        model = load_model(config['model_path'], custom_objects=custom_objects)
        logging.info("Face recognition model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading face recognition model: {str(e)}")
        raise


def detect_and_recognize_faces(image):
    """
    检测和识别图像中的人脸
    """
    try:
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # 获取模型
        model = load_face_model()

        results = []
        for (x, y, w, h) in faces:
            # 提取人脸区域
            face_roi = gray[y:y + h, x:x + w]

            # 调整大小为模型输入大小
            face_roi = cv2.resize(face_roi, (64, 64))

            # 归一化
            face_roi = face_roi.astype('float32') / 255.0

            # 添加通道维度
            face_roi = np.expand_dims(face_roi, axis=-1)

            # 添加批次维度
            face_roi = np.expand_dims(face_roi, axis=0)

            # 进行预测
            prediction = model.predict(face_roi)
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])

            # 添加结果
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'class': config['categories'][class_idx],
                'confidence': confidence
            })

        return results

    except Exception as e:
        logging.error(f"Error in detect_and_recognize_faces: {str(e)}")
        raise


@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    """
    处理人脸识别请求
    """
    try:
        # 验证请求
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 读取和验证图像
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # 检测和识别人脸
        start_time = time.time()
        results = detect_and_recognize_faces(img)
        processing_time = time.time() - start_time

        # 记录结果
        logging.info(f"Processed image with {len(results)} faces detected in {processing_time:.2f} seconds")

        return jsonify({
            'success': True,
            'faces': results,
            'processing_time': processing_time
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




@app.route('/status', methods=['GET'])
def status():
    """
    API状态端点
    """
    return jsonify({
        'status': 'API is running',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    try:
        # 加载配置
        server_config = config.get('server', {})

        # 启动服务器
        app.run(
            host=server_config.get('host', '0.0.0.0'),
            port=server_config.get('port', 5000),
            debug=server_config.get('debug', False)
        )
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
        raise
