import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../combined_results.csv'
df = pd.read_csv(file_path)
df.head()

# X = df[['Group','LSTM','conv','augmented','frame']]
# Y = df[['mAP', 'ECE']]
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
             Group              LSTM              conv            augmented          frame            mAP           ECE      
Group     1.000000e+00,     1.619075e-17,     1.083467e-17,    -7.709882e-18,     9.074034e-17,    -0.200593,    -0.021783
frame     1.619075e-17,     1.000000e+00,    -5.417334e-18,    -4.009139e-17,     1.986356e-17,     0.031799,    -0.046241
LSTM      1.083467e-17,    -5.417334e-18,     1.000000e+00,     5.327045e-17,     2.114711e-17,     0.036204,     0.044185
augment  -7.709882e-18,    -4.009139e-17,     5.327045e-17,     1.000000e+00,    -1.372391e-16,    -0.138181,    -0.156874
conv      9.074034e-17,     1.986356e-17,     2.114711e-17,    -1.372391e-16,     1.000000e+00,     0.030470,    -0.262099
mAP      -2.005928e-01,     3.179882e-02,     3.620435e-02,    -1.381805e-01,     3.047009e-02,     1.000000,    -0.119297
ECE      -2.178255e-02,    -4.624114e-02,     4.418548e-02,    -1.568738e-01,    -2.620994e-01,    -0.119297,     1.000000
"""
print('==============================================================================')

# Group by group usage and calculate mean mAP
mAP_group_Group = df.groupby('Group')['mAP'].mean()
# Group by LSTM usage and calculate mean mAP
mAP_LSTM_Group = df.groupby('LSTM')['mAP'].mean()
# Group by augmented usage and calculate mean mAP
mAP_augmented_Group = df.groupby('augmented')['mAP'].mean()
# Group by conv and calculate mean mAP
mAP_conv_Group = df.groupby('conv')['mAP'].mean()
# Group by frame and calculate mean mAP
mAP_frame_Group = df.groupby('frame')['mAP'].mean()
# Display the reaults
print("mAP_group_Group")
print(mAP_group_Group)
print('==============================================================================')

print("mAP_LSTM_Group")
print(mAP_LSTM_Group)
print('==============================================================================')

print("mAP_augmented_Group")
print(mAP_augmented_Group)
print('==============================================================================')

print("mAP_conv_Group")
print(mAP_conv_Group)
print('==============================================================================')

print("mAP_frame_Group")
print(mAP_frame_Group)
print('==============================================================================')


# Boxplot of mAP by Group usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='mAP', data=df)
plt.title('mAP by Group Usage')
# plt.show()
plt.savefig("mAP_By_Group_Usage.png")

# Boxplot of mAP by LSTM usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='LSTM', y='mAP', data=df)
plt.title('mAP by LSTM Usage')
# plt.show()
plt.savefig("mAP_By_LSTM_Usage.png")

# Boxplot of mAP by augmented usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='augmented', y='mAP', data=df)
plt.title('mAP by augmented Usage')
# plt.show()
plt.savefig("mAP_By_Augmented_Usage.png")

# Boxplot of mAP by conv
plt.figure(figsize=(10, 6))
sns.boxplot(x='conv', y='mAP', data=df)
plt.title('mAP by Convolutional Layer Usage')
# plt.show()
plt.savefig("mAP_By_Convolutional_Layer.png")

# Boxplot of mAP by frame
plt.figure(figsize=(10, 6))
sns.boxplot(x='frame', y='mAP', data=df)
plt.title('mAP by frame')
# plt.show()
plt.savefig("mAP_By_Frame.png")



# Group by group usage and calculate mean mAP
ece_group_Group = df.groupby('Group')['ECE'].mean()
# Group by LSTM usage and calculate mean mAP
ece_LSTM_Group = df.groupby('LSTM')['ECE'].mean()
# Group by augmented usage and calculate mean mAP
ece_augmented_Group = df.groupby('augmented')['ECE'].mean()
# Group by conv and calculate mean mAP
ece_conv_Group = df.groupby('conv')['ECE'].mean()
# Group by frame and calculate mean mAP
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


# Boxplot of mAP by Group usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='ECE', data=df)
plt.title('ECE by Group Usage')
# plt.show()
plt.savefig("ECE_By_Group_Usage.png")

# Boxplot of mAP by LSTM usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='LSTM', y='ECE', data=df)
plt.title('ECE by LSTM Usage')
# plt.show()
plt.savefig("ECE_By_LSTM_Usage.png")

# Boxplot of mAP by augmented usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='augmented', y='ECE', data=df)
plt.title('ECE by augmented Usage')
# plt.show()
plt.savefig("ECE_By_Augmented_Usage.png")

# Boxplot of mAP by conv
plt.figure(figsize=(10, 6))
sns.boxplot(x='conv', y='ECE', data=df)
plt.title('ECE by Convolutional Layer Usage')
# plt.show()
plt.savefig("ECE_By_Convolutional_Layer.png")

# Boxplot of mAP by frame
plt.figure(figsize=(10, 6))
sns.boxplot(x='frame', y='ECE', data=df)
plt.title('ECE by frame')
# plt.show()
plt.savefig("ECE_By_Frame.png")