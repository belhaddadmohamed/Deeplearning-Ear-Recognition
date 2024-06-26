import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a dictionary with the data
data = {
    'Error': ['EER', 'EER', 'EER', 'EER', 'EER', 'FRR', 'FRR', 'FRR', 'FRR', 'FRR', 'FAR', 'FAR', 'FAR', 'FAR', 'FAR'],
    'Classification Method': ['KNN', 'KNN', 'SVM', 'SVM', 'CNN', 'KNN', 'KNN', 'SVM', 'SVM', 'CNN', 'KNN', 'KNN', 'SVM', 'SVM', 'CNN'],
    'Extraction Method': ['ALBP', 'LLBP', 'ALBP', 'LLBP', '-', 'ALBP', 'LLBP', 'ALBP', 'LLBP', '-', 'ALBP', 'LLBP', 'ALBP', 'LLBP', '-'],
    'Value': [7.58, 5.05, 2.27, 0.76, 0.0, 15.0, 10.0, 4.5, 1.5, 0.0, 0.15, 0.10, 0.05, 0.02, 0.0]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Pivot the DataFrame for easier plotting
df_pivot = df.pivot_table(index=['Classification Method', 'Extraction Method'], columns='Error', values='Value').reset_index()

# Define the order of the methods
methods = ['KNN ALBP', 'KNN LLBP', 'SVM ALBP', 'SVM LLBP', 'CNN -']
df_pivot['Method'] = df_pivot['Classification Method'] + ' ' + df_pivot['Extraction Method']
df_pivot = df_pivot.set_index('Method').loc[methods].reset_index()

# Plotting the data
fig, ax = plt.subplots(figsize=(7, 4))

bar_width = 0.2
index = np.arange(len(df_pivot))

# Plot each error type as a group
bars1 = ax.bar(index - bar_width, df_pivot['EER'], bar_width, label='EER', color='b')
bars2 = ax.bar(index, df_pivot['FRR'], bar_width, label='FRR', color='g')
bars3 = ax.bar(index + bar_width, df_pivot['FAR'], bar_width, label='FAR', color='r')

# Add labels and titles
ax.set_xlabel('Methods')
ax.set_ylabel('Percentage')
ax.set_title('Comparison of EER, FRR, and FAR by Classification and Extraction Methods')
ax.set_xticks(index)
ax.set_xticklabels(df_pivot['Method'], rotation=45, ha='right')
ax.legend()

# Add a grid for better readability
ax.yaxis.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
