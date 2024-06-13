import numpy as np
import os
import pandas as pd
import csv
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression

def compute_ece(predictions, labels, num_bins=10):
    """
    Compute the Expected Calibration Error (ECE) for multi-class classification.

    :param predictions: Array of shape (n_samples, n_classes), predicted probabilities.
    :param labels: Array of shape (n_samples,), ground truth labels.
    :param num_bins: Number of bins to use for ECE calculation.
    :return: Expected Calibration Error (ECE), accuracies, confidences, bin_centers.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # Get predicted class and confidence
    predicted_class = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Initialize bins
    accuracies = np.zeros(num_bins)
    confidences_per_bin = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    # Populate bins
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_sizes[i] = np.sum(in_bin)
        if bin_sizes[i] > 0:
            accuracies[i] = np.mean(predicted_class[in_bin] == labels[in_bin])
            confidences_per_bin[i] = np.mean(confidences[in_bin])

    # Calculate ECE
    ece = np.sum((bin_sizes / len(labels)) * np.abs(accuracies - confidences_per_bin))

    return ece, accuracies, confidences_per_bin, bin_centers


def apply_platt_scaling(predictions, labels):
    """
    Apply Platt Scaling to the predicted probabilities.

    :param predictions: Array of shape (n_samples, n_classes), predicted probabilities.
    :param labels: Array of shape (n_samples,), ground truth labels.
    :return: Calibrated predicted probabilities.
    """
    n_classes = predictions.shape[1]
    calibrated_predictions = np.zeros_like(predictions)

    for class_idx in range(n_classes):
        binary_labels = (labels == class_idx).astype(int)
        lr = LogisticRegression()
        lr.fit(predictions[:, class_idx].reshape(-1, 1), binary_labels)
        calibrated_predictions[:, class_idx] = lr.predict_proba(predictions[:, class_idx].reshape(-1, 1))[:, 1]

    return calibrated_predictions / calibrated_predictions.sum(axis=1, keepdims=True)


def apply_isotonic_regression(predictions, labels):
    """
    Apply Isotonic Regression to the predicted probabilities.

    :param predictions: Array of shape (n_samples, n_classes), predicted probabilities.
    :param labels: Array of shape (n_samples,), ground truth labels.
    :return: Calibrated predicted probabilities.
    """
    n_classes = predictions.shape[1]
    calibrated_predictions = np.zeros_like(predictions)

    for class_idx in range(n_classes):
        binary_labels = (labels == class_idx).astype(int)
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(predictions[:, class_idx], binary_labels)
        calibrated_predictions[:, class_idx] = iso_reg.transform(predictions[:, class_idx])

    return calibrated_predictions / calibrated_predictions.sum(axis=1, keepdims=True)


# 작업 디렉토리 설정
work_dir = '../work_dir'  # 실제 작업 디렉토리로 변경하세요

# 결과를 저장할 데이터프레임 초기화
result_df = pd.DataFrame(columns=['model_name', 'augmented', 'frame', 'loss', 'accuracy'])
ece_list = [["Model", "ECE", "Group", "LSTM", "conv", "augmented", "frame", "mAP", "normal_acc", "mandown_acc", "cross_acc"]]

platt_scaling_ece = []
isotonic_regression_ece = []
original_ece = []

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
                h5_parts = file_name.split('_')
                name = f"./plots/{parts[1]}_{parts[2]}_{h5_parts[2].split('.')[0]}.png"
                ece = compute_ece(multi_probs, labels)

                # ECE 계산 및 그래프 시각화 (원래 예측 확률)
                ece_original, acc_original, conf_original, bin_centers = compute_ece(multi_probs, labels)
                print(f'Expected Calibration Error (Original): {ece_original:.4f}')

                # ECE 계산 및 그래프 시각화 (Platt Scaling 적용)
                calibrated_predictions = apply_platt_scaling(multi_probs, labels)
                ece_platt, acc_platt, conf_platt, _ = compute_ece(calibrated_predictions, labels)
                print(f'Expected Calibration Error (Platt Scaling): {ece_platt:.4f}')

                # ECE 계산 및 그래프 시각화 (isotonic regression 적용)
                calibrated_predictions = apply_isotonic_regression(multi_probs, labels)
                ece_isot, acc_isot, conf_isot, _ = compute_ece(calibrated_predictions, labels)
                print(f'Expected Calibration Error (Isotonic regression): {ece_isot:.4f}')

                # Plot calibration
                plt.figure(figsize=(10, 6))
                plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

                # 원래 예측 확률에 대한 그래프
                plt.plot(bin_centers, acc_original, 's-', label='Original Accuracy', color='blue')
                # plt.plot(bin_centers, conf_original, 's-', label='Original Confidence', color='red')

                # Platt Scaling 적용 후 예측 확률에 대한 그래프
                plt.plot(bin_centers, acc_platt, 'o-', label='Platt Scaled Accuracy', color='magenta')
                # plt.plot(bin_centers, conf_platt, 'o-', label='Platt Scaled Confidence', color='magenta')

                plt.fill_between(bin_centers, acc_original, bin_centers, color='gray', alpha=0.3,
                                 label='Original Gap')
                plt.fill_between(bin_centers, acc_platt, bin_centers, color='gray', alpha=0.1, label='Platt Scaled Gap')

                plt.xlabel('Confidence')
                plt.ylabel('Accuracy')
                plt.title(f'Calibration Plot ECE: {ece_original*1000}, Platt Scaled ECE: {ece_platt*1000}')
                plt.legend()
                plt.grid(True)
                # plt.savefig(name)
                plt.close()

                platt_scaling_ece.append(ece_platt)
                original_ece.append(ece_original)
                isotonic_regression_ece.append(ece_isot)

ece_data = {
    'Original_ECE': original_ece,
    'Platt_scaling_ECE': platt_scaling_ece,
    'Isotonic_regression_ECE': isotonic_regression_ece
}

df = pd.DataFrame(ece_data)
df.to_csv('ece_comparison.csv', index=False)
