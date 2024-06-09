import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# 샘플 데이터: 각 클래스에 대한 예측 결과
y_true = np.array([0, 1, 1, 0, 1, 0, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.7])

# Precision-Recall 곡선 계산
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# AP 계산
average_precision = average_precision_score(y_true, y_scores)

print("Precision: ", precision)
print("Recall: ", recall)
print("Average Precision (AP): ", average_precision)
