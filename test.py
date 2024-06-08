import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset import HOFS_E
import numpy as np
import os
import re
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def list_h5_files(directory):
    output = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.h5'):
                output.append(str(os.path.join(root, name)))
    return output

def test_model(model_path, dataset, group=False, n_category=3):
    model_path = f"work_dir/{model_path}"
    model_list = list_h5_files(model_path)
    result = [['frame', 'loss', 'accuracy']]

    for model_name in model_list:
        print('model : ', model_name)
        # Load the trained model
        model = load_model(model_name)
        frame = int(re.findall(r'\d+', model_name)[-2])

        # Define the path to your test CSV file
        test_csv_path = f'{dataset}/HOFS_E_{frame}_test.csv'

        if group == True:
            # Initialize the dataset instance with the test CSV path
            dataset_instance = HOFS_E(test_csv_path, frame, 1, False)

            # Get the test dataset
            test_dataset = dataset_instance.get_dataset()
            test_head = np.array(test_dataset[5])
            test_right_arm = np.array(test_dataset[1])
            test_left_arm = np.array(test_dataset[2])
            test_upper_body = np.array(test_dataset[3])
            test_lower_body = np.array(test_dataset[4])
            test_whole_body = np.array(test_dataset[0])
            test_label = np.array(test_dataset[6])
            test_label = tf.keras.utils.to_categorical(test_label, num_classes=n_category)

            # Evaluate the model on the test data
            results = model.evaluate([test_whole_body, test_left_arm, test_right_arm, test_upper_body, test_lower_body, test_head],
                                     test_label, batch_size=16)

            print("Test loss:", results[0])
            print("Test accuracy:", results[1])
            result_list = [frame, results[0], results[1]]
            result.append(result_list)

            predictions = model.predict([test_whole_body, test_left_arm, test_right_arm, test_upper_body, test_lower_body, test_head])


        elif group == False:
            # Initialize the dataset instance with the test CSV path
            dataset_instance = HOFS_E(test_csv_path, frame, 1, False)

            # Get the test dataset
            test_dataset = dataset_instance.get_dataset()
            test_whole_body = np.array(test_dataset[0])
            test_label = np.array(test_dataset[6])
            test_label = tf.keras.utils.to_categorical(test_label, num_classes=n_category)

            # Evaluate the model on the test data
            results = model.evaluate([test_whole_body],
                                     test_label, batch_size=16)

            print("Test loss:", results[0])
            print("Test accuracy:", results[1])
            result_list = [frame, results[0], results[1]]
            result.append(result_list)

            predictions = model.predict([test_whole_body])

        predicted_classes = np.argmax(predictions, axis=1)

        # 실제 클래스 라벨을 one-hot encoding에서 정수 인덱스로 변환
        actual_classes = np.argmax(test_label, axis=1)

        for i in range(len(predicted_classes)):
            print(f"Sample {i}: Actual class {actual_classes[i]}, Predicted class {predicted_classes[i]}")

        # 실제 클래스와 예측된 클래스를 사용하여 혼동 행렬 생성
        cm = confusion_matrix(actual_classes, predicted_classes)

        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 20})  # annot_kws로 주석 크기 설정
        plt.title(f'{model_name} Confusion Matrix', size=16)  # 제목의 글꼴 크기 설정
        plt.ylabel('Actual class', size=14)  # y축 레이블의 글꼴 크기 설정
        plt.xlabel('Predicted class', size=14)  # x축 레이블의 글꼴 크기 설정
        plt.xticks(fontsize=12)  # x축 눈금 레이블의 글꼴 크기 설정
        plt.yticks(fontsize=12)  # y축 눈금 레이블의 글꼴 크기 설정

        # Save the plot as a file
        plot_filename = os.path.join(model_path,
                                     f'test_{os.path.basename(model_name)}_confusion_matrix.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory


        print("Test loss:", results[0])
        print("Test accuracy:", results[1])
        result_list = [frame, results[0], results[1]]
        result.append(result_list)


    with open(f'{model_path}/result.csv', 'w', newline ='') as file:
        writer = csv.writer(file)
        writer.writerows(result)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with different configurations.")
    parser.add_argument("--model_path", type=str, required=True, help="model path")
    parser.add_argument("--dataset", type=str, required=True, help="Select test dataset in data folder (ex.'csves_diff')")
    parser.add_argument("--group", type=lambda x:(True if x=='True'else(False if x=='False' else argparse.ArgumentTypeError('Boolean value expected.'))), required=True, help="Data augmentation")
    parser.add_argument("--n_category", type=int, required=False, help="number of category")

    args = parser.parse_args()

    model_path = args.model_path
    dataset = args.dataset
    group = args.group
    n_category = args.n_category

    test_model(model_path, dataset, group, n_category)