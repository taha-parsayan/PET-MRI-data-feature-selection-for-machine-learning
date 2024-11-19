
#%% Imports
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import levene
import matplotlib.pyplot as plt





#%% Import features from the excel file

path = 'Features-2.xlsx'
AD_image_features = pd.read_excel(path, sheet_name='AD')
AD_SUVR_ref = pd.read_excel(path, sheet_name='AD-ref')
AD_medical_features = pd.read_excel(path, sheet_name='AD-medical')

MCI_image_features = pd.read_excel(path, sheet_name='MCI')
MCI_SUVR_ref = pd.read_excel(path, sheet_name='MCI-ref')
MCI_medical_features = pd.read_excel(path, sheet_name='MCI-medical')

NC_image_features = pd.read_excel(path, sheet_name='NC')
NC_SUVR_ref = pd.read_excel(path, sheet_name='NC-ref')
NC_medical_features = pd.read_excel(path, sheet_name='NC-medical')


# Calculate SUVR
for i in range (120):
    AD_image_features.iloc[i, 2:117] /= AD_SUVR_ref.iloc[i,0]
    MCI_image_features.iloc[i, 2:117] /= MCI_SUVR_ref.iloc[i,0]
    NC_image_features.iloc[i, 2:117] /= NC_SUVR_ref.iloc[i,0]

    
# Put all features together
AD = pd.concat([AD_image_features, AD_medical_features.iloc[:,1:]], axis=1)
MCI = pd.concat([MCI_image_features, MCI_medical_features.iloc[:,1:]], axis=1)
NC = pd.concat([NC_image_features, NC_medical_features.iloc[:,1:]], axis=1)

# Male:1 , Female:0
gender_map = {'M': 1, 'F': 0}
AD['Sex'] = AD['Sex'].replace(gender_map)
MCI['Sex'] = MCI['Sex'].replace(gender_map)
NC['Sex'] = NC['Sex'].replace(gender_map)


'''
col 0 -> Subjects
col 1 -> Group
col 2:20 -> subcortical SUVR
col 21:116 -> cortical SUVR
col 117:135 -> subcortical volume
col 136:231 -> cortical volume
col 232 -> MHPSYCH
col 233 -> MH2NEURL
col 234 -> MH4CARD
col 235 -> MMSCORE
col 236 -> CLINICAL DEMENTIA RATING
'''


# Mean & SD
AD_mean = AD.mean(numeric_only=True)
MCI_mean = MCI.mean(numeric_only=True)
NC_mean = NC.mean(numeric_only=True)

AD_SD = AD.std(numeric_only=True)
MCI_SD = MCI.std(numeric_only=True)
NC_SD = NC.std(numeric_only=True)





#%% Outlier handling

def replace_outliers_with_median(df, columns):
    for col in columns:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with the mean or median (change below if you want median)
        df[col] = df[col].apply(lambda x: np.median(df[col]) if x < lower_bound or x > upper_bound else x)
    return df




collumns_to_check = AD.columns[2:232]
AD = replace_outliers_with_median(AD, collumns_to_check)
MCI = replace_outliers_with_median(MCI, collumns_to_check)
NC = replace_outliers_with_median(NC, collumns_to_check)





#%% initiations

results_suvr_all = pd.DataFrame(AD.columns[2:117], columns=['ROI'])
results_suvr_AD_vs_MCI = []
results_suvr_MCI_vs_NC = []

results_volume_all = pd.DataFrame(AD.columns[117:232], columns=['ROI'])
results_volume_AD_vs_MCI = []
results_volume_MCI_vs_NC = []




# %% AD vs. MCI (SUVR)
num_of_ROIs = 0
for i in range(2, 117):
    group1 = AD.iloc[:,i]
    group2 = MCI.iloc[:,i]
    corrected_type_I_error = 0.05/115
    
    # equality of variance
    stat, p_value = levene(group1, group2)
    if p_value < corrected_type_I_error:
        var_equal = False
    else:
        var_equal = True

    # two sample t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var = var_equal)
    if p_value < corrected_type_I_error:
        print(AD.columns[i])
        results_suvr_AD_vs_MCI.append(1)
        num_of_ROIs += 1
    else:
        results_suvr_AD_vs_MCI.append(0)

print(f"Number of ROIs: {num_of_ROIs}")
results_suvr_AD_vs_MCI = pd.DataFrame(results_suvr_AD_vs_MCI, columns=['suvr_AD_vs_MCI'])
results_suvr_all['suvr_AD_vs_MCI'] = results_suvr_AD_vs_MCI


# %% MCI vs. NC (SUVR)
num_of_ROIs = 0
for i in range(2, 117):
    group1 = MCI.iloc[:,i]
    group2 = NC.iloc[:,i]
    corrected_type_I_error = 0.05/115
    
    # equality of variance
    stat, p_value = levene(group1, group2)
    if p_value < corrected_type_I_error:
        var_equal = False
    else:
        var_equal = True

    # two sample t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var = var_equal)
    if p_value < corrected_type_I_error:
        print(AD.columns[i])
        results_suvr_MCI_vs_NC.append(1)
        num_of_ROIs += 1
    else:
        results_suvr_MCI_vs_NC.append(0)

print(f"Number of ROIs: {num_of_ROIs}")
results_suvr_MCI_vs_NC = pd.DataFrame(results_suvr_MCI_vs_NC, columns=['suvr_MCI_vs_NC'])
results_suvr_all['suvr_MCI_vs_NC'] = results_suvr_MCI_vs_NC



# %% AD vs. MCI (volume)
num_of_ROIs = 0
for i in range(117, 232):
    group1 = AD.iloc[:,i]
    group2 = MCI.iloc[:,i]
    corrected_type_I_error = 0.05/115
    
    # equality of variance
    stat, p_value = levene(group1, group2)
    if p_value < corrected_type_I_error:
        var_equal = False
    else:
        var_equal = True

    # two sample t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var = var_equal)
    if p_value < corrected_type_I_error:
        print(AD.columns[i])
        results_volume_AD_vs_MCI.append(1)
        num_of_ROIs += 1
    else:
        results_volume_AD_vs_MCI.append(0)

print(f"Number of ROIs: {num_of_ROIs}")
results_volume_AD_vs_MCI = pd.DataFrame(results_volume_AD_vs_MCI, columns=['volume_AD_vs_MCI'])
results_volume_all['volume_AD_vs_MCI'] = results_volume_AD_vs_MCI


# %% MCI vs. NC (volume)
num_of_ROIs = 0
for i in range(117, 232):
    group1 = MCI.iloc[:,i]
    group2 = NC.iloc[:,i]
    corrected_type_I_error = 0.05/115
    
    # equality of variance
    stat, p_value = levene(group1, group2)
    if p_value < corrected_type_I_error:
        var_equal = False
    else:
        var_equal = True

    # two sample t-test
    t_stat, p_value = ttest_ind(group1, group2, equal_var = var_equal)
    if p_value < corrected_type_I_error:
        print(AD.columns[i])
        results_volume_MCI_vs_NC.append(1)
        num_of_ROIs += 1
    else:
        results_volume_MCI_vs_NC.append(0)

print(f"Number of ROIs: {num_of_ROIs}")
results_volume_MCI_vs_NC = pd.DataFrame(results_volume_MCI_vs_NC, columns=['volume_MCI_vs_NC'])
results_volume_all['volume_MCI_vs_NC'] = results_volume_MCI_vs_NC







# %% Plot

temp1 = pd.DataFrame(AD.mean(numeric_only=True))
temp2 = pd.DataFrame(MCI.mean(numeric_only=True))
temp3 = pd.DataFrame(NC.mean(numeric_only=True))

#drop left and right whole crebral cortex for the sake of plotting
temp1 = temp1.drop('1_left_cerebral_cortex_SUVR,')
temp1 = temp1.drop('11_right_cerebral_cortex_SUVR,')
temp1 = temp1.drop('11_right_cerebral_cortex_volume,')
temp1 = temp1.drop('1_left_cerebral_cortex_volume,')
temp2 = temp2.drop('1_left_cerebral_cortex_SUVR,')
temp2 = temp2.drop('11_right_cerebral_cortex_SUVR,')
temp2 = temp2.drop('11_right_cerebral_cortex_volume,')
temp2 = temp2.drop('1_left_cerebral_cortex_volume,')
temp3 = temp3.drop('1_left_cerebral_cortex_SUVR,')
temp3 = temp3.drop('11_right_cerebral_cortex_SUVR,')
temp3 = temp3.drop('11_right_cerebral_cortex_volume,')
temp3 = temp3.drop('1_left_cerebral_cortex_volume,')


# Define larger font sizes
plt.rcParams.update({
    'font.size': 14,        # General font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # Axes labels font size
    'xtick.labelsize': 12,  # X-axis tick labels font size
    'ytick.labelsize': 12,  # Y-axis tick labels font size
    'legend.fontsize': 12   # Legend font size
})

# Create subplots for better layout
fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

# Plot SUVR Mean Comparison (No lines, only markers)
axes[0].plot(
    temp1.iloc[0:113], label='AD', color='red', marker='o', linestyle='None', markersize=6, alpha=0.8
)
axes[0].plot(
    temp2.iloc[0:113], label='MCI', color='orange', marker='s', linestyle='None', markersize=6, alpha=0.8
)
axes[0].plot(
    temp3.iloc[0:113], label='NC', color='green', marker='^', linestyle='None', markersize=6, alpha=0.8
)
axes[0].set_xlabel('ROI')
axes[0].set_ylabel('Average SUVR')
#axes[0].set_title('Average SUVR Comparison')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# Plot Volume Mean Comparison (No lines, only markers)
axes[1].plot(
    temp1.iloc[115:228], label='AD', color='red', marker='o', linestyle='None', markersize=6, alpha=0.8
)
axes[1].plot(
    temp2.iloc[115:228], label='MCI', color='orange', marker='s', linestyle='None', markersize=6, alpha=0.8
)
axes[1].plot(
    temp3.iloc[115:228], label='NC', color='green', marker='^', linestyle='None', markersize=6, alpha=0.8
)
axes[1].set_xlabel('ROI')
axes[1].set_ylabel('Average Volume (mm$^3$)')
#axes[1].set_title('Average Volume Comparison')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

# Show the final plots
plt.show()


# %%
