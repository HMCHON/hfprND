import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

data_file_path = '5-10_combined_results.csv'

data_list = []

with open(data_file_path, newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data_list.append(row)

conv_list = []
frame_list = []
accuracy_list = []
augmented_list = []

for row in data_list[1:]:
    # Data augmentation
    if any(model in row for model in ['A']):
        augmented = 'A'
        augmented_list.append('A')
    elif any(model in row for model in ['NA']):
        augmented = 'NA'
        augmented_list.append('NA')

    # Convolution layer
    if any(model in row[0] for model in ['model10', 'model16', 'model22', 'model28']):
        conv = 6
        conv_list.append(conv)
    elif any(model in row[0] for model in ['model9', 'model15', 'model21', 'model27']):
        conv = 5
        conv_list.append(conv)
    elif any(model in row[0] for model in ['model8', 'model14', 'model20', 'model26']):
        conv = 4
        conv_list.append(conv)
    elif any(model in row[0] for model in ['model7', 'model13', 'model19', 'model25']):
        conv = 3
        conv_list.append(conv)
    elif any(model in row[0] for model in ['model6', 'model12', 'model18', 'model24']):
        conv = 2
        conv_list.append(conv)
    elif any(model in row[0] for model in ['model5', 'model11', 'model17', 'model23']):
        conv = 1
        conv_list.append(int(conv))

    # frame
    frame = row[2]
    # accuracy
    accuracy = row[4]

    frame_list.append(int(frame))
    accuracy_list.append(float(accuracy) * 100)

# 3D plot 그리기
fig = plt.figure(figsize=(12, 8), dpi=300)  # 그림 크기를 12x8 인치로 설정
ax = fig.add_subplot(111, projection='3d')

# 색상 설정
colors = ['orange' if augmented == 'A' else 'blue' for augmented in augmented_list]

# 산점도
scatter = ax.scatter(conv_list, frame_list, accuracy_list, c=colors)

# 각 점에 레이블 추가
for i in range(len(conv_list)):
    ax.text(conv_list[i], frame_list[i], accuracy_list[i],
            f'Acc:{accuracy_list[i]:.2f}\n'
            f'Conv:{conv_list[i]}\n'
            f'Fr:{frame_list[i]}\n',
            color='black', fontsize=3)

# 축 레이블 설정
ax.set_xlabel('conv')
ax.set_ylabel('frame')
ax.set_zlabel('accuracy')


# 범례 추가
orange_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='A')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='NA')
ax.legend(handles=[orange_patch, blue_patch])

plt.savefig(f'{data_file_path}_test_3D.png')
plt.show()
