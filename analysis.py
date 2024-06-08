import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'combined_results1.csv'
df = pd.read_csv(file_path)
df.head()

X = df[['Group','LSTM','conv','augmented','frame']]
Y = df['accuracy']

model = sm.OLS(Y,X).fit()

print(model.summary())
print('==============================================================================')

# Basic statistical summary
summary_stats = df.describe()
print(summary_stats)
print('==============================================================================')

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
"""
                Group            LSTM            conv          augmented         frame       accuracy       
 Group       1.000000e+00,   -7.709882e-17,   -3.936596e-16,   2.004569e-17,   -8.848312e-17,   -0.065000
 frame      -7.709882e-17,    1.000000e+00,   -1.047351e-16,  -9.868649e-17,   -4.333867e-17,   -0.046270
 LSTM       -3.936596e-16,   -1.047351e-16,    1.000000e+00,  -2.979534e-17,   -9.146123e-17,   -0.028995
 augmented   2.004569e-17,    -9.868649e-17,  -2.979534e-17,   1.000000e+00,   -2.166933e-17,   -0.021867
 conv       -8.848312e-17,   -4.333867e-17,   -9.146123e-17,  -2.166933e-17,    1.000000e+00,    0.014138
 accuracy   -6.499983e-02,   -4.626967e-02,   -2.899483e-02,  -2.186741e-02,    1.413817e-02,    1.000000
"""
print('==============================================================================')

# Group by group usage and calculate mean accuracy
group_Group = df.groupby('Group')['accuracy'].mean()
# Group by LSTM usage and calculate mean accuracy
LSTM_Group = df.groupby('LSTM')['accuracy'].mean()
# Group by augmented usage and calculate mean accuracy
augmented_Group = df.groupby('augmented')['accuracy'].mean()
# Group by conv and calculate mean accuracy
conv_Group = df.groupby('conv')['accuracy'].mean()
# Group by frame and calculate mean accuracy
frame_Group = df.groupby('frame')['accuracy'].mean()
# Display the reaults
print("group_Group")
print(group_Group)
print('==============================================================================')

print("LSTM_Group")
print(LSTM_Group)
print('==============================================================================')

print("augmented_Group")
print(augmented_Group)
print('==============================================================================')

print("conv_Group")
print(conv_Group)
print('==============================================================================')

print("frame_Group")
print(frame_Group)
print('==============================================================================')


# Boxplot of accuracy by Group usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='accuracy', data=df)
plt.title('Accuracy by Group Usage')
# plt.show()
plt.savefig("Accuracy_By_Group_Usage.png")

# Boxplot of accuracy by LSTM usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='LSTM', y='accuracy', data=df)
plt.title('Accuracy by LSTM Usage')
# plt.show()
plt.savefig("Accuracy_By_LSTM_Usage.png")

# Boxplot of accuracy by augmented usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='augmented', y='accuracy', data=df)
plt.title('Accuracy by augmented Usage')
# plt.show()
plt.savefig("Accuracy_By_Augmented_Usage.png")

# Boxplot of accuracy by conv
plt.figure(figsize=(10, 6))
sns.boxplot(x='conv', y='accuracy', data=df)
plt.title('Accuracy by Convolutional Layer Usage')
# plt.show()
plt.savefig("Accuracy_By_Convolutional_Layer.png")

# Boxplot of accuracy by frame
plt.figure(figsize=(10, 6))
sns.boxplot(x='frame', y='accuracy', data=df)
plt.title('Accuracy by frame')
# plt.show()
plt.savefig("Accuracy_By_Frame.png")
