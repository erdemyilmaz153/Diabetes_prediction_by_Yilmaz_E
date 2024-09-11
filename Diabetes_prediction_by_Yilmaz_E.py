# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

matplotlib.use('QT5Agg')
from matplotlib import pyplot
from matplotlib import pyplot as plt

# To see all columns at once
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

#####################################################################################################################
###################################### Load and Examine the Dataset #################################################
#####################################################################################################################

df = pd.read_csv('diabetes.csv')
'''
Information about dataset attributes -
Pregnancies: To express the Number of pregnancies
Glucose: To express the Glucose level in blood
BloodPressure: To express the Blood pressure measurement
SkinThickness: To express the thickness of the skin
Insulin: To express the Insulin level in blood
BMI: To express the Body mass index
DiabetesPedigreeFunction: To express the Diabetes percentage
Age: To express the age
Outcome: To express the final result 1 is Yes and 0 is No
'''

def check_data(dataframe,head=5):
    print(20*"-" + "Information".center(20) + 20*"-")
    print(dataframe.info())
    print(20*"-" + "Data Shape".center(20) + 20*"-")
    print(dataframe.shape)
    print("\n" + 20*"-" + "The First 5 Data".center(20) + 20*"-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print(dataframe.isnull().sum())
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)

check_data(df)
'''
768 rows and 9 columns - one('Outcome') of the columns is the target variable.

There is no missing data but 0 values.

All of the variables, except BMI and DiabetesPedigreeFunction being float64, are int64.

Glucose, BloodPressure, SKinThickness, Insulin, and BMI have zero values which is not possible under standard
conditions.

Insulin has high std and maximum value as 846.000 meaning that it has high interval between low and high values that
might indicate outliers or significant variability in the data. The large different between its mean(79.799) and 50th
percentile (30.500) indicated positive skewness. The distribution possibly has a long right tail, with a few very 
high values skewing the average.

BMI having zeros and relatively high std (7.884) suggests some degree of skewness or the presence of outliers.

Diabetes Pedigree function has a maximum value of 2.420 which is significantly higher than the mean (0.472) and 99th
percentile (1.698) that suggest a long right tail in the distribution.

The outcome has a mean of 0.349 that indicates of 34.9% of the individuals being diabetes in the dataset. It suggests
that the dataset is imbalanced.
'''


#######################################################################################################################

# Plot histograms for each feature
df.hist(bins=20, figsize=(15, 10))
plt.show()
'''
Histogram graphs give us some idea here however 0 values need to be handled before jumping into conclusions.
'''


#####################################################################################################################
###################################### Handling 0(zero) values ######################################################
#####################################################################################################################

# Count rows where each individual column is 0
glucose_zero_count = (df['Glucose'] == 0).sum()
blood_pressure_zero_count = (df['BloodPressure'] == 0).sum()
skin_thickness_zero_count = (df['SkinThickness'] == 0).sum()
insulin_zero_count = (df['Insulin'] == 0).sum()
bmi_zero_count = (df['BMI'] == 0).sum()

# Report the counts
print(f"Number of rows where Glucose is 0: {glucose_zero_count}")   # 5
print(f"Number of rows where BloodPressure is 0: {blood_pressure_zero_count}")   # 35
print(f"Number of rows where SkinThickness is 0: {skin_thickness_zero_count}")   # 227
print(f"Number of rows where Insulin is 0: {insulin_zero_count}")   # 374
print(f"Number of rows where BMI is 0: {bmi_zero_count}")   # 11


#####################################################################################################################

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Correlation of all the variables with just the target
# Compute the correlation matrix
corr_matrix = df.corr()

# Extract correlations with the target variable 'MEDV'
target_corr = corr_matrix['Outcome'].sort_values(ascending=False)

# Display the correlations with the target variable
print(target_corr)

'''
Age - Pregnancies
Age - Glucose
SkinThickness - BMI
SkinThickness - Insulin
Outcome - Glucose

The pairs above have high and positive correlation between them. This can be useful while handling with 0 values.

'''

#####################################################################################################################

'''
The 0 values for glucose, and BMI is relatively low, they are directly deleted.
In addition, after correlation heatmap checked, blood pressure has very low correlation with the outcome and relatively
low with others, and has relatively low number as 0, the rows of it is directly deleted as well.
'''

# Drop rows where Glucose is 0
df = df[df['Glucose'] != 0]

# Drop rows where BloodPressure is 0
df = df[df['BloodPressure'] != 0]

# Drop rows where BMI is 0
df = df[df['BMI'] != 0]

# Print the shape of the DataFrame after dropping the rows
print(f"Shape of DataFrame after dropping rows with Glucose, BloodPressure, or BMI being 0: {df.shape}")   # (724, 9)


#####################################################################################################################

# Scan through several rows to examine 0 value in the columns
df.head(30)
df.tail(25)

# Drop rows where both SkinThickness and Insulin are 0
df = df[~((df['SkinThickness'] == 0) & (df['Insulin'] == 0))]

# Print the shape of the DataFrame after dropping the rows
print(f"Shape of DataFrame after dropping rows with both SkinThickness and Insulin being 0: {df.shape}")   # (532, 9)


#####################################################################################################################

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Correlation of all the variables with just the target
# Compute the correlation matrix
corr_matrix = df.corr()

# Extract correlations with the target variable 'MEDV'
target_corr = corr_matrix['Outcome'].sort_values(ascending=False)

# Display the correlations with the target variable
print(target_corr)

'''
Now, the correlations became more clear.
The correlations mentioned above has now higher positive value except the correlation between Insulin and SkinThickness
got smaller.
'''


#####################################################################################################################
###################################### Outlier Handling #############################################################
#####################################################################################################################

# Box-and-whiskers plots of the variables to handle with outliers
for column in df.columns:
    plt.figure(figsize=(10, 5))
    df.boxplot(column=column, vert=False)
    plt.title(f'Box-and-Whisker Plot for {column}')
    plt.show()

'''
Glucose has no outliers.
Pregnancies, SkinThickness, and BMI has low # of outliers above 3rd interquartile(IQR) threshold.
BloodPressure has low # of outliers below 1st IQR threshold, and also above 3rd IQR threshold.
In order to handle outliers of 'Age' and 'Insulin', they is going to be categorized according to literature.
DiabetesPedigreeFunction is also going to be categorized since cumulative number of observations are seen around 3rd 
IQR.
'''


###################################################################################################################

# Since outliers above 3rd Interquartile(IQR) threshold for the columns below has not that much effect, they are
#replaced with 3rd IQR.
# Function to replace outliers with Q3
def replace_outliers_with_q3(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: Q3 if x > threshold else x)

# Apply the function to the specified columns
columns_to_replace = ['Pregnancies', 'SkinThickness', 'BMI']
for column in columns_to_replace:
    replace_outliers_with_q3(df, column)

# Check the result
print(df)

# Function to replace outliers with Q1 or Q3
def replace_outliers_blood_pressure(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR
    df[column] = df[column].apply(lambda x: Q1 if x < lower_threshold else (Q3 if x > upper_threshold else x))

# Apply the function to BloodPressure column
replace_outliers_blood_pressure(df, 'BloodPressure')

# Check the result
print(df)


####################################################################################################################

# According to literature, age is categorized as following:
# Function to categorize age
def categorize_age(age):
    if age < 30:
        return '18-29'
    elif age < 45:
        return '30-44'
    elif age < 60:
        return '45-59'
    else:
        return '60+'

# Apply the function to create a new 'AgeCategory' column
df['AgeCategory'] = df['Age'].apply(categorize_age)

# Check the result
print(df)


####################################################################################################################

# Also, insulin is categorized as in the literature.
# Function to categorize insulin levels
def categorize_insulin(insulin):
    if insulin < 360:
        return 'Below 360'
    else:
        return 'Above 360'

# Apply the function to create a new 'InsulinCategory' column
df['InsulinCategory'] = df['Insulin'].apply(categorize_insulin)

# Check the result
print(df)


####################################################################################################################

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)

# Create a correlation matrix for numerical columns
correlation_matrix = df[numerical_cols].corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Numerical Columns')
plt.show()


####################################################################################################################

# Apply logarithmic transformation with handling for skewness, additionally for zero or negative values
df['Insulin_log'] = np.log1p(df['Insulin'])
df['Age_log'] = np.log(df['Age'])
df['DiabetesPedigreeFunction_log'] = np.log1p(df['DiabetesPedigreeFunction'])

# Check the transformed DataFrame
print(df)


#####################################################################################################################
###################################### Feature Engineering ##########################################################
#####################################################################################################################

# According to literature, DiabetesPedigree function is positively correlated with age.
df['DiabetesPedigreeFunction_times_Age'] = df['DiabetesPedigreeFunction'] * df['Age']

# Since glucose has correlated with diabetes, it effect is increased by taking square of it.
df['Glucose_squared'] = df['Glucose']**2

# According to importance at the end, the effect of glucose and BMI on the model is high. To increase the impact,
#and fact that they are related, they are divided with each other.
df['Glucose_times_BMI'] = df['Glucose'] * df['BMI']

# With simple logic, we can conclude that pregnancies increases by age.
df['Pregnancies_divided_by_Age'] = df['Pregnancies'] / df['Age']

# Since these has still high skewness after even the logarithmic transformations, they are eliminated and when the
#performance check, it is more beneficial.
df = df.drop(['Insulin', 'Age', 'DiabetesPedigreeFunction'], axis=1)
df.head()


#####################################################################################################################
####################################### Encoding ####################################################################
#####################################################################################################################

# Get categorical columns to apply encoding.
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Check the result
print(df_encoded)
df.head()
df = df_encoded

#####################################################################################################################
###################################### Training the Model ###########################################################
#####################################################################################################################

# Step 1: Define features (X) and target (y)
X = df.drop(columns='Outcome')  # All columns except the target
y = df['Outcome']  # The target variable

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Create and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")   # 0.83
print("Classification Report:")   # F1 = 0.88 for 0 --- F1 = 0.72 for 1. It is expected since suport is 74 and 33,
# respectively.

print(report)


####################################################################################################################

# Step 7: Calculate and display feature importances (coefficients)
feature_importances = model.coef_[0]
features = X.columns

# Combine feature names with their importance scores
importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': feature_importances
})

# Sort the features by the absolute value of the coefficient
importance_df['Absolute Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

print("\nFeature Importances:")
print(importance_df[['Feature', 'Coefficient']])

# Step 8: Visualize the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances (Coefficients) from Logistic Regression')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

'''
We safely say that our complete processing of the data has increased performance of the model after examining the 
importance graph.
'''
###########################################

# Step 5: Get predicted probabilities on the test set
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Step 6: Apply a different threshold
threshold = 0.48  # Example threshold
y_pred = (y_prob >= threshold).astype(int)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)



####################################
################################
######### SMOTE TRY

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Step 1: Create and train the logistic regression model
clf = LogisticRegression(random_state=888)
clf.fit(X_train_scaled, y_train)

# Step 2: Predict probabilities on the test set
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class

# Step 3: Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"ROC AUC Score: {roc_auc}")

# Step 3: Generate the classification report
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)



from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Step 1: Define features (X) and target (y)
X = df.drop(columns='Outcome')  # All columns except the target
y = df['Outcome']  # The target variable

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE to the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Apply scaling to the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create and train the logistic regression model
clf = LogisticRegression(random_state=888)
clf.fit(X_train_scaled, y_train_resampled)

# Step 6: Predict the class labels on the test set
y_pred = clf.predict(X_test_scaled)

# Step 7: Generate the classification report
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)
































