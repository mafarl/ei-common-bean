import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data (specify the file paths, models are all available for particular problem type)
# CLASSIFICATION
file_paths = [
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_AdaBoostClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_AutoKeras_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_AutoLGBM_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_AutoSKLearn_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_AutoXGBoost_49.csv",
    "../../experiments/results/gh_new_img_autoxai2/results/scores_CV_FixedKeras_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_GaussianProcessClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_GradientBoostingClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_KNeighborsClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_MLPClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_RandomForestClassifier_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_SVC_49.csv",
    "../../experiments/results/multiclass_ps_4classes_new_img/results/scores_CV_XGBClassifier_49.csv"
]

# REGRESSION
"""
file_paths = [
    "../../experiments/results/flower_win23_new_image/results/scores_CV_AdaBoostRegressor_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_AutoKeras_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_AutoLGBM_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_AutoSKLearn_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_AutoXGBoost_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_DecisionTreeRegressor_49.csv",
    "../../experiments/results/drought_e_apparent/results/scores_CV_FixedKeras_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_Ridge_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_GradientBoostingRegressor_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_KNeighborsRegressor_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_LinearRegression_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_RandomForestRegressor_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_SVR_49.csv",
    "../../experiments/results/flower_win23_new_image/results/scores_CV_XGBRegressor_49.csv",
]
"""

# Read DataFrames including GBLUP file
df_gblup = pd.read_csv("../../gblup_multiclass_4classes/GAPIT.Cross.Validation_2_113_gBLUP.txt", delimiter='\t')
# Only read second column that contains cross validation results
dataframes = [pd.read_csv(file).iloc[:, 1] for file in file_paths]

# Remove unneeded columns from GBLUP
df_gblup = df_gblup.drop(["Reference_2_113_gBLUP", "All_2_113_gBLUP"], axis=1)
df_gblup = df_gblup.reset_index(drop=True)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes, axis=1)

# Rename the columns
# REGRESSION PLOTS
#combined_df.columns = ['AdaBoostRegres', 'AutoKeras', 'AutoLGBM', 'AutoSKLearn', 'AutoXGBoost', 'DecisionTreeRegres', 'FixedKeras', 'Ridge', 'GradBoostRegres', 'KNRegres', 'LinearRegres', 'RFRegres', 'SVR', 'XGBRegres']
# CLASSIFICATION PLOTS
combined_df.columns = ['AdaBoostClass', 'AutoKeras', 'AutoLGBM', 'AutoSKLearn', 'AutoXGBoost', 'FixedKeras', 'GaussianClass', 'GradBoostClass', 'KNClass', 'MLPClass', 'RFClass', 'SVC', 'XGBClass']
# Add GBLUP to the concantenated DataFrame
combined_df['GBLUP'] = df_gblup

# Plot violion plot
plt.figure(figsize=(22, 8))
sns.violinplot(data=combined_df, palette="hls", width=0.8, scale='width', cut=0)
plt.title('Violin Plot of Cross Validation Results')
plt.savefig('multi_4classes/violin_plot.png')

# Plot box plot (with dots for values)
plt.figure(figsize=(20, 8))
ax = sns.stripplot(data=combined_df)
sns.boxplot(data=combined_df, palette="hls", saturation=0.4, ax=ax)
ax.set_ylim(0,1.01)
plt.title('Box Plot of Cross Validation Results')
plt.savefig('multi_4classes/box_plot.png')
