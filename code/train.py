import os
import cv2
import numpy as np
from keras.src.layers import BatchNormalization
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.regularizers import l2

def load_and_preprocess_dataset(data_dir, image_size=(64, 64)):
    """
    加载数据集并进行预处理。

    :param data_dir: 数据集目录
    :param image_size: 图像大小，默认为 (64, 64)
    :return: 图像数据和标签
    """
    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory.")
        return None, None, None

    imgs = []
    labs = []

    # 假设所有图像都属于同一个类别
    class_num = 0

    for img_name in os.listdir(data_dir):
        try:
            img_path = os.path.join(data_dir, img_name)
            # 读取图像并转换为灰度图像
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                print(f"Error reading image {img_name}: Image is None")
                continue

            img_array = cv2.resize(img_array, image_size)
            imgs.append(img_array)
            labs.append(class_num)

        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

    imgs = np.array(imgs).reshape(-1, image_size[0], image_size[1], 1)
    labs = np.array(labs)

    # 将标签转换为二分类格式
    labs = to_categorical(labs, num_classes=2)

    # 将图片数据归一化到 [0,1] 范围
    imgs = imgs / 255.0

    return imgs, labs, ['class_0', 'class_1']

def build_and_train_model(X_train, y_train, X_test, y_test, num_classes, epochs=10):
    """
    构建并训练卷积神经网络模型。

    :param X_train: 训练集图像数据
    :param y_train: 训练集标签
    :param X_test: 测试集图像数据
    :param y_test: 测试集标签
    :param num_classes: 类别数量
    :param epochs: 训练轮数，默认为 10
    :return: 训练好的模型
    """
    model = Sequential()

    # 添加 L2 正则化和 BatchNormalization
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:], kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 使用 TensorBoard 实时查看模型训练进度
    tensorboard = TensorBoard(log_dir='./logs')

    # 使用数据增强技术
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    datagen.fit(X_train)

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard])

    return model

if __name__ == "__main__":
    data_dir = 'output_faces'
    image_size = (64, 64)

    # 加载并预处理数据集
    imgs, labs, categories = load_and_preprocess_dataset(data_dir, image_size)

    if imgs is None or labs is None or categories is None:
        print("Error: Failed to load and preprocess dataset.")
    else:
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(imgs, labs, test_size=0.2, random_state=42)

        # 构建并训练模型
        num_classes = len(categories)
        model = build_and_train_model(X_train, y_train, X_test, y_test, num_classes)

        # 评估模型性能
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')

        # 保存模型
        model.save('face_recognition_model.h5')
