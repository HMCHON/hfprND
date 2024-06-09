import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'combined_results.csv'
df = pd.read_csv(file_path)
df.head()

# X = df[['Group','LSTM','conv','augmented','frame']]
# Y = df[['accuracy', 'ECE']]
#
# model = sm.OLS(Y,X).fit()
#
# print(model.summary())
# print('==============================================================================')

# Basic statistical summary
summary_stats = df.describe()
print(summary_stats)
print('==============================================================================')

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)
"""
            Group              LSTM              conv            augmented          frame          accuracy         ECE      
Group    1.000000e+00,     3.546546e-17,    -3.069822e-17,     3.083953e-17,    -6.229934e-17,    -0.065000,    -0.021783
frame    3.546546e-17,     1.000000e+00,    -3.250400e-17,     2.467162e-17,     9.028889e-18,    -0.046270,    -0.046241
LSTM    -3.069822e-17,    -3.250400e-17,     1.000000e+00,     3.566411e-17,    -1.564886e-16,    -0.028995,     0.044185
augment  3.083953e-17,     2.467162e-17,     3.566411e-17,     1.000000e+00,     1.616171e-16,    -0.021867,    -0.156874
conv    -6.229934e-17,     9.028889e-18,    -1.564886e-16,     1.616171e-16,     1.000000e+00,     0.014138,    -0.262099
acc     -6.499983e-02,    -4.626967e-02,    -2.899483e-02,    -2.186741e-02,     1.413817e-02,     1.000000,    -0.252464
ECE     -2.178255e-02,    -4.624114e-02,     4.418548e-02,    -1.568738e-01,    -2.620994e-01,    -0.252464,     1.000000
"""
print('==============================================================================')

# Group by group usage and calculate mean accuracy
acc_group_Group = df.groupby('Group')['accuracy'].mean()
# Group by LSTM usage and calculate mean accuracy
acc_LSTM_Group = df.groupby('LSTM')['accuracy'].mean()
# Group by augmented usage and calculate mean accuracy
acc_augmented_Group = df.groupby('augmented')['accuracy'].mean()
# Group by conv and calculate mean accuracy
acc_conv_Group = df.groupby('conv')['accuracy'].mean()
# Group by frame and calculate mean accuracy
acc_frame_Group = df.groupby('frame')['accuracy'].mean()
# Display the reaults
print("acc_group_Group")
print(acc_group_Group)
print('==============================================================================')

print("acc_LSTM_Group")
print(acc_LSTM_Group)
print('==============================================================================')

print("acc_augmented_Group")
print(acc_augmented_Group)
print('==============================================================================')

print("acc_conv_Group")
print(acc_conv_Group)
print('==============================================================================')

print("acc_frame_Group")
print(acc_frame_Group)
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



# Group by group usage and calculate mean accuracy
ece_group_Group = df.groupby('Group')['ECE'].mean()
# Group by LSTM usage and calculate mean accuracy
ece_LSTM_Group = df.groupby('LSTM')['ECE'].mean()
# Group by augmented usage and calculate mean accuracy
ece_augmented_Group = df.groupby('augmented')['ECE'].mean()
# Group by conv and calculate mean accuracy
ece_conv_Group = df.groupby('conv')['ECE'].mean()
# Group by frame and calculate mean accuracy
ece_frame_Group = df.groupby('frame')['ECE'].mean()
# Display the reaults
print("ece_group_Group")
print(ece_group_Group)
print('==============================================================================')

print("ece_LSTM_Group")
print(ece_LSTM_Group)
print('==============================================================================')

print("ece_augmented_Group")
print(ece_augmented_Group)
print('==============================================================================')

print("ece_conv_Group")
print(ece_conv_Group)
print('==============================================================================')

print("ece_frame_Group")
print(ece_frame_Group)
print('==============================================================================')


# Boxplot of accuracy by Group usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='ECE', data=df)
plt.title('ECE by Group Usage')
# plt.show()
plt.savefig("ECE_By_Group_Usage.png")

# Boxplot of accuracy by LSTM usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='LSTM', y='ECE', data=df)
plt.title('ECE by LSTM Usage')
# plt.show()
plt.savefig("ECE_By_LSTM_Usage.png")

# Boxplot of accuracy by augmented usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='augmented', y='ECE', data=df)
plt.title('ECE by augmented Usage')
# plt.show()
plt.savefig("ECE_By_Augmented_Usage.png")

# Boxplot of accuracy by conv
plt.figure(figsize=(10, 6))
sns.boxplot(x='conv', y='ECE', data=df)
plt.title('ECE by Convolutional Layer Usage')
# plt.show()
plt.savefig("ECE_By_Convolutional_Layer.png")

# Boxplot of accuracy by frame
plt.figure(figsize=(10, 6))
sns.boxplot(x='frame', y='ECE', data=df)
plt.title('ECE by frame')
# plt.show()
plt.savefig("ECE_By_Frame.png")