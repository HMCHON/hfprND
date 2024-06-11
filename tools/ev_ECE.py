import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def expected_calibration_error(probs, labels, num_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    probs (np.ndarray): Predicted probabilities for the positive class.
    labels (np.ndarray): True binary labels (0 or 1).
    num_bins (int): Number of bins to use for calibration.

    Returns:
    float: Expected Calibration Error.
    """
    # Initialize the bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Calculate ECE
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of the predictions that fall into the current bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


# 작업 디렉토리 설정
work_dir = '../work_dir'  # 실제 작업 디렉토리로 변경하세요

# 결과를 저장할 데이터프레임 초기화
result_df = pd.DataFrame(columns=['model_name', 'augmented', 'frame', 'loss', 'accuracy'])
ece_list = [["Model", "ECE", "Group", "LSTM", "conv", "augmented", "frame", "mAP", "normal_acc", "mandown_acc", "cross_acc"]]

# 작업 디렉토리 내 모든 폴더 순회
for folder_name in os.listdir(work_dir):
    folder_path = os.path.join(work_dir, folder_name)
    if os.path.isdir(folder_path):
        # 폴더 이름에서 model_name과 augmented 추출
        parts = folder_name.split('_')
        model_name = parts[0] + '_' + parts[1]
        augmented = parts[1]

        # 폴더 내 CSV 파일 읽기
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.h5'):
                pre_csv_name = f'{file_name}_pre_result.csv'
                file_path = os.path.join(folder_path, pre_csv_name)
                df = pd.read_csv(file_path)

                # 필요한 열만 추출하여 새로운 데이터프레임 생성
                df = df[['actual class', 'predicted class', 'normal', 'mandown', 'cross']]

                probs = []
                labels = []
                multi_probs = []
                normal_acc = []
                mandown_acc = []
                cross_acc = []

                for index, row in df.iterrows():
                    actual_class = int(row['actual class'])
                    predicted_class = int(row['predicted class'])
                    normal = row['normal']
                    mandown = row['mandown']
                    cross = row['cross']
                    probablity = [normal, mandown, cross]
                    prob = probablity[actual_class]

                    probs.append(prob)
                    labels.append(actual_class)
                    multi_probs.append(probablity)

                probs = np.array(probs)
                labels = np.array(labels)
                multi_probs = np.array(multi_probs)
                ece = expected_calibration_error(probs, labels)
                print(f'Model: {folder_path}\{file_name}, Expected Calibration Error (ECE): {ece}')

                y_true_binary = label_binarize(labels, classes=[0, 1, 2])
                average_precisions = []
                for i in range(3):
                    precision, recall, _ = precision_recall_curve(y_true_binary[:, i], multi_probs[:, i])
                    average_precision = average_precision_score(y_true_binary[:, i], multi_probs[:, i])
                    average_precisions.append(average_precision)
                    print(f"Model: {folder_path}\{file_name}, Class {i} - AP: {average_precision}")
                    if i == 0:
                        normal_acc = average_precision
                    elif i == 1:
                        mandown_acc = average_precision
                    elif i == 2:
                        cross_acc = average_precision

                # mAP 계산
                mAP = np.mean(average_precisions)
                print(f"Model: {folder_path}\{file_name}, Mean Average Precision (mAP): {mAP}")


                parts = folder_name.split('_')
                if parts[2] == 'A':
                    augment = 1
                elif parts[2] == 'NA':
                    augment = 0

                h5_parts = file_name.split('_')
                frame = int(h5_parts[2].split('.')[0])

                # Group
                if parts[1] in ["model5", "model6", "model7", "model8", "model9", "model10", "model11", "model12", "model13", "model14", "model15", "model16"]:
                    group = 0
                else:
                    group = 1

                # LSTM
                if parts[1] in ["model11", "model12", "model13", "model14", "model15", "model16", "model23", "model24", "model25", "model26", "model27", "model28"]:
                    LSTM = 1
                else:
                    LSTM = 0

                # conv
                if parts[1] in ["model5", "model11", "model17", "model23"]:
                    conv = 1
                elif parts[1] in ["model6", "model12", "model18", "model24"]:
                    conv = 2
                elif parts[1] in ["model7", "model13", "model19", "model25"]:
                    conv = 3
                elif parts[1] in ["model8", "model14", "model20", "model26"]:
                    conv = 4
                elif parts[1] in ["model9", "model15", "model21", "model27"]:
                    conv = 5
                elif parts[1] in ["model10", "model16", "model22", "model28"]:
                    conv = 6


                ece_list.append([parts[1], ece, group, LSTM, conv, augment, frame, mAP, normal_acc, mandown_acc, cross_acc])

with open('../check_ece_result.csv', 'w', newline ='') as file:
    writer = csv.writer(file)
    writer.writerows(ece_list)
