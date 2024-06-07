import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'combined_result.csv'
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
 Group       1.000000e+00,    3.854941e-17,  -5.913923e-17,   5.396917e-18,   1.236958e-16,  -0.077577
 frame       3.854941e-17,    1.000000e+00,   1.444622e-17,   3.777842e-17,   1.805778e-17,  -0.076148
 LSTM       -5.913923e-17,    1.444622e-17,   1.000000e+00,   4.965889e-18,  -1.913813e-16,   0.013839
 augmented   5.396917e-18,    3.777842e-17,   4.965889e-18,   1.000000e+00,  -1.480738e-16,   0.035470
 conv        1.236958e-16,    1.805778e-17,  -1.913813e-16,  -1.480738e-16,   1.000000e+00,  -0.052751
 accuracy   -7.757747e-02,  - 7.614803e-02,   1.383894e-02,   3.547049e-02,  -5.275072e-02,   1.000000
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
plt.show()

# Boxplot of accuracy by LSTM usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='LSTM', y='accuracy', data=df)
plt.title('Accuracy by LSTM Usage')
plt.show()

# Boxplot of accuracy by augmented usage
plt.figure(figsize=(10, 6))
sns.boxplot(x='augmented', y='accuracy', data=df)
plt.title('Accuracy by augmented Usage')
plt.show()

# Boxplot of accuracy by conv
plt.figure(figsize=(10, 6))
sns.boxplot(x='conv', y='accuracy', data=df)
plt.title('Accuracy by Convolutional Layer Usage')
plt.show()

# Boxplot of accuracy by frame
plt.figure(figsize=(10, 6))
sns.boxplot(x='frame', y='accuracy', data=df)
plt.title('Accuracy by frame')
plt.show()