!pip install catboost
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
df = pd.read_csv('dataset_traffic_accident_prediction1.csv')
df.describe()
df.info()
df.isnull().sum()

columns_with_missing = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Alcohol',
                       'Driver_Age', 'Driver_Experience',
                       'Accident']

for i in columns_with_missing:
    print(f"{i} - Среднее: {df[i].mean()}, Медиана: {df[i].median()}")
    df[i].plot(kind='hist', bins=30, title=i)
    plt.axvline(df[i].mean(), color='red', linestyle='dashed', linewidth=2)
    plt.axvline(df[i].median(), color='blue', linestyle='dashed', linewidth=2)
    plt.show()


    df['Traffic_Density'].fillna(df['Traffic_Density'].median(), inplace=True)
df['Speed_Limit'].fillna(df['Speed_Limit'].median(), inplace=True)
df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].median(), inplace=True)
df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].median(), inplace=True)
df['Driver_Age'].fillna(df['Driver_Age'].median(), inplace=True)
df['Driver_Experience'].fillna(df['Driver_Experience'].median(), inplace=True)
df['Accident'].fillna(df['Accident'].median(), inplace=True)

columns_miss = df.columns[df.isnull().sum() > 0]

df[columns_miss].info()

df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
df['Road_Type'] = df['Road_Type'].fillna(df['Road_Type'].mode()[0])
df['Time_of_Day'] = df['Time_of_Day'].fillna(df['Time_of_Day'].mode()[0])
df['Accident_Severity'] = df['Accident_Severity'].fillna(df['Accident_Severity'].mode()[0])
df['Road_Condition'] = df['Road_Condition'].fillna(df['Road_Condition'].mode()[0])
df['Vehicle_Type'] = df['Vehicle_Type'].fillna(df['Vehicle_Type'].mode()[0])
df['Road_Light_Condition'] = df['Road_Light_Condition'].fillna(df['Road_Light_Condition'].mode()[0])

df.isnull().sum()

duplicates_count = df.duplicated().sum()
print(f"Duplication: {duplicates_count}")

df = df.drop_duplicates()
duplicates_count1 = df.duplicated().sum()
print(f"Duplication: {duplicates_count1}")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

sns.boxplot(x=df['Speed_Limit'], ax=axes[0, 0])
axes[0, 0].set_title('Speed_Limit')

sns.boxplot(x=df['Number_of_Vehicles'], ax=axes[0, 1])
axes[0, 1].set_title('Number_of_Vehicles')

sns.boxplot(x=df['Driver_Age'], ax=axes[1, 0])
axes[1, 0].set_title('Driver_Age')

sns.boxplot(x=df['Driver_Experience'], ax=axes[1, 1])
axes[1, 1].set_title('Driver_Experience')

sns.boxplot(x=df['Traffic_Density'], ax=axes[2, 0])
axes[2, 0].set_title('Traffic_Density')

fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.show()

cols = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')

numeric_df = df.select_dtypes(include=['number'])
spearman_corr = numeric_df.corr(method='spearman')

print(spearman_corr)

plt.figure(figsize=(8, 7))
sns.heatmap(spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Spearman Correlation Matrix')
plt.show()

df = pd.get_dummies(df, columns=['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 
                                  'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'], drop_first=True)

df['Age_vs_Experience'] = df['Driver_Age'] - df['Driver_Experience']
df = df.drop(['Driver_Age', 'Driver_Experience'], axis = 1)

X = df.drop(['Accident'], axis = 1)
y = df['Accident']

feature_names = X.columns

numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Age_vs_Experience']

scaler = RobustScaler()

X_scaled = X.copy()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.33, random_state=100) 

xgb_classifier = xgb.XGBClassifier(eval_metric='mlogloss')

param_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}


grid_search__xgb = GridSearchCV(xgb_classifier, param_xgb, cv=5)

grid_search__xgb.fit(X_train, y_train)


grid_search__xgb.best_params_

best_gs_xgb = grid_search__xgb.best_estimator_

print('Score on train data = ', round(best_gs_xgb.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_xgb.score(X_test, y_test), 4))


rf_classifier = RandomForestClassifier()

parametrs_fr = {'max_depth':[3, 5, 10],
                'n_estimators':[150, 200, 300],
                'min_samples_split': [2, 5, 10],  
                'min_samples_leaf': [1, 2, 4]}

grid_search_reg_forest3 = GridSearchCV(rf_classifier, parametrs_fr, cv = 5)

grid_search_reg_forest3.fit(X_train, y_train)

grid_search_reg_forest3.best_params_

best_gs_rf = grid_search_reg_forest3.best_estimator_

print('Score on train data = ', round(best_gs_rf.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_rf.score(X_test, y_test), 4))

knn_classifier = KNeighborsClassifier()

param_knn = {
    'n_neighbors': range(1, 30),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'leaf_size': range(10, 50, 5)}

grid_search_knn = GridSearchCV(knn_classifier, param_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

grid_search_knn.best_params_

best_gs_knn = grid_search_knn.best_estimator_

print('Score on train data = ', round(best_gs_knn.score(X_train, y_train), 4))
print('Score on test data = ', round(best_gs_knn.score(X_test, y_test), 4))


#deployment
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# Load and preprocess dataset (reuse your existing processing code here)
# Ensure 'df', 'scaler', 'feature_names', 'numeric_columns', and 'best_gs_xgb' are available

# Only keep what's needed to transform new input
scaler = RobustScaler()
X_scaled = X.copy()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Define prediction function
def predict_accident(Speed_Limit, Number_of_Vehicles, Traffic_Density, Driver_Alcohol,
                     Weather, Road_Type, Time_of_Day, Accident_Severity, Road_Condition,
                     Vehicle_Type, Road_Light_Condition, Driver_Age, Driver_Experience):
    
    Age_vs_Experience = Driver_Age - Driver_Experience

    input_data = {
        'Speed_Limit': Speed_Limit,
        'Number_of_Vehicles': Number_of_Vehicles,
        'Traffic_Density': Traffic_Density,
        'Driver_Alcohol': Driver_Alcohol,
        'Age_vs_Experience': Age_vs_Experience,
        **{f"Weather_{Weather}": 1},
        **{f"Road_Type_{Road_Type}": 1},
        **{f"Time_of_Day_{Time_of_Day}": 1},
        **{f"Accident_Severity_{Accident_Severity}": 1},
        **{f"Road_Condition_{Road_Condition}": 1},
        **{f"Vehicle_Type_{Vehicle_Type}": 1},
        **{f"Road_Light_Condition_{Road_Light_Condition}": 1},
    }

    # Fill missing dummy vars with 0
    for col in feature_names:
        input_data.setdefault(col, 0)

    input_df = pd.DataFrame([input_data])
    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])
    
    prediction = best_gs_xgb.predict(input_df)[0]
    return "Accident" if prediction == 1 else "No Accident"

# Build the interface
demo = gr.Interface(
    fn=predict_accident,
    inputs=[
        gr.Number(label="Speed Limit"),
        gr.Number(label="Number of Vehicles"),
        gr.Number(label="Traffic Density"),
        gr.Number(label="Driver Alcohol Level"),
        gr.Dropdown(choices=df['Weather'].unique(), label="Weather"),
        gr.Dropdown(choices=df['Road_Type'].unique(), label="Road Type"),
        gr.Dropdown(choices=df['Time_of_Day'].unique(), label="Time of Day"),
        gr.Dropdown(choices=df['Accident_Severity'].unique(), label="Accident Severity"),
        gr.Dropdown(choices=df['Road_Condition'].unique(), label="Road Condition"),
        gr.Dropdown(choices=df['Vehicle_Type'].unique(), label="Vehicle Type"),
        gr.Dropdown(choices=df['Road_Light_Condition'].unique(), label="Road Light Condition"),
        gr.Number(label="Driver Age"),
        gr.Number(label="Driver Experience"),
    ],
    outputs="text",
    title="Traffic Accident Prediction",
    description="Predict whether an accident will occur based on input parameters."
)

demo.launch()
