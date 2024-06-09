import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tr_dataset import HOFS_E
from models.model3 import model3
from models.model4 import model4
from models.model5 import model5
from models.model6 import model6
from models.model7 import model7
from models.model8 import model8
from models.model9 import model9
from models.model10 import model10
from models.model11 import model11
from models.model12 import model12
from models.model13 import model13
from models.model14 import model14
from models.model15 import model15
from models.model16 import model16
from models.model17 import model17
from models.model18 import model18
from models.model19 import model19
from models.model20 import model20
from models.model21 import model21
from models.model22 import model22
from models.model23 import model23
from models.model24 import model24
from models.model25 import model25
from models.model26 import model26
from models.model27 import model27
from models.model28 import model28

import numpy as np
import os

def train_model3(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model3(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model4(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model4(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model5(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model5(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model6(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model6(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model7(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model7(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model8(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model8(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model9(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model9(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model10(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model10(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model11(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model11(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model12(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model12(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model13(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model13(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model14(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model14(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model15(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model15(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model16(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model16(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body],
                         val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback])


def train_model17(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model17(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model18(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model18(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=([val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model19(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model19(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model20(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model20(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model21(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model21(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model22(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model22(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model23(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model23(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model24(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model24(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model25(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model25(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model26(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model26(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model27(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model27(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


def train_model28(n_frames, dataset, save_folder_name, n_category=3, augment=False, sigma=1):
    # 모델 인스턴스 생성
    input_shape1 = (n_frames, 34)  # n_frames, n_joints
    model_instance = model28(input_shape1, n_category)
    model = model_instance.get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Define Training data
    dataset_instance = HOFS_E(f"{dataset}/HOFS_E_{n_frames}_train.csv", n_frames, sigma, augment)
    train_dataset = dataset_instance.get_dataset()
    train_head = np.array(train_dataset[5])
    train_right_arm = np.array(train_dataset[1])
    train_left_arm = np.array(train_dataset[2])
    train_upper_body = np.array(train_dataset[3])
    train_lower_body = np.array(train_dataset[4])
    train_whole_body = np.array(train_dataset[0])
    train_label = np.array(train_dataset[6])
    train_label = to_categorical(train_label, num_classes=n_category)

    # Define validation data
    dataset_instance = HOFS_E(f'{dataset}/HOFS_E_{n_frames}_val.csv', n_frames, sigma, augment)
    val_dataset = dataset_instance.get_dataset()
    val_head = np.array(val_dataset[5])
    val_right_arm = np.array(val_dataset[1])
    val_left_arm = np.array(val_dataset[2])
    val_upper_body = np.array(val_dataset[3])
    val_lower_body = np.array(val_dataset[4])
    val_whole_body = np.array(val_dataset[0])
    val_label = np.array(val_dataset[6])
    val_label = to_categorical(val_label, num_classes=3)

    # 모델 학습
    if not os.path.exists(f"work_dir/{save_folder_name}"):
        path = f"work_dir/{save_folder_name}"
        os.makedirs(path)
    checkpoint = ModelCheckpoint(f'work_dir/{save_folder_name}/model_best_{n_frames}.h5', monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='loss', patience=50)
    tensorboard_callback = TensorBoard(log_dir=f'work_dir/{save_folder_name}/model_best_{n_frames}', histogram_freq=1)

    history = model.fit(
        [train_whole_body, train_left_arm, train_right_arm, train_upper_body, train_lower_body, train_head],
        train_label,
        epochs=1000,
        batch_size=16,
        validation_data=(
        [val_whole_body, val_left_arm, val_right_arm, val_upper_body, val_lower_body, val_head], val_label),
        callbacks=[checkpoint, earlystopping, tensorboard_callback]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with different configurations.")
    parser.add_argument("--n_frames", type=int, required=True, help="Number of frames (1 or 3).")
    parser.add_argument("--model", type=str, required=True, help="Select train model in model folder (ex.'model3')")
    parser.add_argument("--dataset", type=str, required=True, help="Select train dataset in data folder (ex.'csves_diff')")
    parser.add_argument("--save_folder_name", type=str, required=True, help="Folder name bout to save the trained model.")
    parser.add_argument("--n_category", type=int, required=False, help="Number of class", default=3)
    parser.add_argument("--augment", type=lambda x:(True if x=='True'else(False if x=='False' else argparse.ArgumentTypeError('Boolean value expected.'))), required=False, help="data augmentation", default=False)
    parser.add_argument("--sigma", type=int, required=False, help="Gaussian distribution sigma for data augmentation", default=1)
    args = parser.parse_args()

    n_frames = args.n_frames
    model = args.model
    dataset = args.dataset
    save_folder_name = args.save_folder_name
    n_category = args.n_category
    augment = args.augment
    sigma = args.sigma

    if model == "model3":
        train_model3(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model4":
        train_model4(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model5":
        train_model5(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model6":
        train_model6(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model7":
        train_model7(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model8":
        train_model8(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model9":
        train_model9(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model10":
        train_model10(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model11":
        train_model11(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model12":
        train_model12(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model13":
        train_model13(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model14":
        train_model14(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model15":
        train_model15(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model16":
        train_model16(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model17":
        train_model17(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model18":
        train_model18(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model19":
        train_model19(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model20":
        train_model20(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model21":
        train_model21(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model22":
        train_model22(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model23":
        train_model23(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model24":
        train_model24(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model25":
        train_model25(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model26":
        train_model26(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model27":
        train_model27(n_frames, dataset, save_folder_name, n_category, augment, sigma)
    elif model == "model28":
        train_model28(n_frames, dataset, save_folder_name, n_category, augment, sigma)
