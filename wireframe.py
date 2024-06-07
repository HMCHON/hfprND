# import numpy as np
# import pandas as pd
# from scipy.interpolate import make_interp_spline
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 원본 데이터 준비
# conv_list = [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
# frame_list = [1, 5, 7, 3, 11, 9, 1, 5, 7, 3, 11, 9, 1, 5, 7, 3, 11, 9, 1, 5, 7, 3, 11, 9, 1, 5, 7, 3, 11, 9, 1, 5, 7, 3, 11, 9]
# accuracy_list = [92.79379250000001, 94.98886466026306, 93.19196343421936, 95.2222228, 89.01345133781433, 92.95302033424376, 81.48558735847473, 94.09799575805664, 93.41517686843872, 94.44444179534912, 92.93721914291382, 93.28858852386476, 94.4567621, 94.5434272, 94.75446343421936, 94.77777481079102, 93.3856487, 93.17673444747923, 78.27051281929016, 93.76391768455504, 91.62946343421936, 94.44444179534912, 91.70403480529784, 94.07159090042114, 91.90687537193298, 93.98663640022278, 93.41517686843872, 95.66666483879088, 88.90134692192078, 89.82102870941162, 88.69179487228394, 94.76614594459534, 91.74107313156128, 94.9999988, 92.82511472702026, 92.39373803138731]
#
# # conv=6인 데이터만 추출 및 재정렬
# conv_6_indices = [i for i, x in enumerate(conv_list) if x == 6]
# frame_6_sorted = [1, 3, 5, 7, 9, 11]
# accuracy_6_sorted = [accuracy_list[i] for i in conv_6_indices]
#
# # 나머지 데이터 추출
# conv_other = [conv_list[i] for i in range(len(conv_list)) if conv_list[i] != 6]
# frame_other = [frame_list[i] for i in range(len(frame_list)) if conv_list[i] != 6]
# accuracy_other = [accuracy_list[i] for i in range(len(accuracy_list)) if conv_list[i] != 6]
#
# # 재정렬된 데이터와 나머지 데이터를 합치기
# conv_combined = [6]*6 + conv_other
# frame_combined = frame_6_sorted + frame_other
# accuracy_combined = accuracy_6_sorted + accuracy_other
#
# # Interpolating data
# conv_array = np.array(conv_combined)
# frame_array = np.array(frame_combined)
# accuracy_array = np.array(accuracy_combined)
#
# # Creating spline functions
# conv_spline = make_interp_spline(np.arange(len(conv_array)), conv_array, k=3)
# frame_spline = make_interp_spline(np.arange(len(frame_array)), frame_array, k=3)
# accuracy_spline = make_interp_spline(np.arange(len(accuracy_array)), accuracy_array, k=3)
#
# # Generating new data points
# xnew = np.linspace(0, len(conv_array) - 1, 100)
# conv_smooth = conv_spline(xnew)
# frame_smooth = frame_spline(xnew)
# accuracy_smooth = accuracy_spline(xnew)
#
# # Creating 3D plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Surface plot
# ax.plot_trisurf(conv_smooth, frame_smooth, accuracy_smooth, color='lightblue', alpha=0.6)
#
# # Scatter plot
# ax.scatter(conv_list, frame_list, accuracy_list, color='red')
#
# # Labels
# ax.set_xlabel('Conv')
# ax.set_ylabel('Frame')
# ax.set_zlabel('Accuracy')
#
# plt.show()



import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 예시 데이터 생성
data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 3, 4, 5, 6],
    'X3': [5, 4, 3, 2, 1],
    'X4': [2, 2, 2, 2, 2],
    'X5': [1, 3, 5, 7, 9],
    'BoolVar': [True, False, True, False, True],
    'Y': [10, 12, 14, 16, 18]
}

df = pd.DataFrame(data)

# boolean 변수를 0과 1로 변환
df['BoolVar'] = df['BoolVar'].astype(int)

# 독립 변수와 종속 변수 분리
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'BoolVar']]
Y = df['Y']

# 상수항 추가 (절편을 위해)
X = sm.add_constant(X)

# 회귀 모델 피팅
model = sm.OLS(Y, X).fit()

# 결과 출력
print(model.summary())

# 회귀 분석 결과 시각화
# 변수 하나를 선택하여 시각화 (예: X1)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['X1'], y=Y, color='blue', label='Actual Data')
sns.lineplot(x=df['X1'], y=model.predict(X), color='red', label='Regression Line')
plt.xlabel('X1')
plt.ylabel('Y')
plt.title('Scatter plot of X1 vs Y with Regression Line')
plt.legend()
plt.show()

# 변수 간의 상관 관계 히트맵
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()