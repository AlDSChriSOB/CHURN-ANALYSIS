#!/usr/bin/env python
# coding: utf-8

# ![churn.jpeg](attachment:churn.jpeg)

# 
# ## Background
# ----
# <span style="font-size: larger;">
#     
#     
# **ConnectTel** is a leading telecommunications company at the forefront of innovation and connectivity solutions. With a strong presence in the global market, ConnectTel has established itself as a trusted provider of reliable voice, data, and Internet services. 
# 
# Offering a comprehensive range of telecommunications solutions, including mobile networks, broadband connections, and enterprise solutions, ConnectTel caters to both individual and corporate customers, they are committed to providing exceptional customer service and cutting-edge technology.ConnectTel ensures seamless communication experiences for millions of users worldwide. 
# 
# Through strategic partnerships and a customer-centric approach, ConnectTel continues to revolutionize the telecom industry, empowering individuals and businesses to stay connected and thrive in the digital age.</span> 

# 
# ## Problem Statement
# ----
# <span style="font-size: larger;">
# 
# 
# 
# **ConnectTel Telecom Company** faces the pressing need to address
# customer churn, which poses a significant threat to its business
# sustainability and growth.
# 
# The company's current customer retention strategies lack precision and
# effectiveness, resulting in the loss of valuable customers to competitors.
# To overcome this challenge, **ConnectTel** aims to develop a
# robust customer churn prediction system for which you have been contacted
# to handle as a Data Scientist. By leveraging advanced analytics and machine
# learning techniques on available customer data, the company seeks to
# accurately forecast customer churn and implement targeted retention
# initiatives.
# 
# This proactive approach will enable ConnectTel to reduce customer
# attrition, enhance customer loyalty, and maintain a competitive edge in the
# highly dynamic and competitive telecommunications industry. </span> 

# ## Data Dictionary
# ----
# 
# <div style="line-height: 2;">
# 1. CustomerID: A unique identifier assigned to each telecom customer, enabling
# tracking and identification of individual customers.
# 
# 2. Gender: The gender of the customer, which can be categorized as male, or
# female. This information helps in analyzing gender-based trends in
# customer churn.
# 
# 3. SeniorCitizen: A binary indicator that identifies whether the customer is a senior citizen
# or not. This attribute helps in understanding if there are any specific
# churn patterns among senior customers.
# 
# 4. Partner: Indicates whether the customer has a partner or not. This attribute helps
# in evaluating the impact of having a partner on churn behavior.
# 
# 5. Dependents: Indicates whether the customer has dependents or not. This attribute
# helps in assessing the influence of having dependents on customer
# churn.
# 
# 6. Tenure: The duration for which the customer has been subscribed to the telecom
# service. It represents the loyalty or longevity of the customer’s
# relationship with the company and is a significant predictor of churn.
# 
# 7. PhoneService: Indicates whether the customer has a phone service or not. This attribute
# helps in understanding the impact of phone service on churn.
# 
# 8. MultipleLines: Indicates whether the customer has multiple lines or not. This attribute helps in analyzing
# the effect of having multiple lines on customer churn.
# 
# 9. InternetService: Indicates the type of internet service subscribed by the customer, such as DSL, fiber optic,
# or no internet service. It helps in evaluating the relationship between internet service and churn.
# 
# 10. OnlineSecurity: Indicates whether the customer has online security services or not. This attribute helps in
# analyzing the impact of online security on customer churn.
# 
# 11. OnlineBackup: Indicates whether the customer has online backup services or not. This attribute helps in
# evaluating the impact of online backup on churn behavior.
# 
# 12. DeviceProtection: Indicates whether the customer has device protection services or not. This attribute helps
# in understanding the influence of device protection on churn.
# 
# 13. TechSupport: Indicates whether the customer has technical support services or not. This attribute helps
# in assessing the impact of tech support on churn behavior.
# 
# 14. StreamingTV: Indicates whether the customer has streaming TV services or not. This attribute helps in
# evaluating the impact of streaming TV on customer churn.
# 
# 15. StreamingMovies: Indicates whether the customer has streaming movie services or not. This attribute helps in understanding the influence of streaming movies on churn behavior.
# 
# 16. Contract: Indicates the type of contract the customer has, such as a month-to-month, one-year, or two-year contract. It is a crucial factor in predicting churn as different contract lengths may have varying impacts on customer loyalty.
# 
# 17. PaperlessBilling: Indicates whether the customer has opted for paperless billing or not. This attribute helps in analyzing the effect of paperless billing on customer churn.
# 
# 18. PaymentMethod: Indicates the method of payment used by the customer, such as electronic checks, mailed checks, bank transfers, or credit cards. This attribute helps in evaluating the impact of payment methods on churn.
# 
# 19. MonthlyCharges: The amount charged to the customer on a monthly basis. It helps in understanding the relationship between monthly charges and churn behavior.
# 
# 20. TotalCharges: The total amount charged to the customer over the entire tenure. It represents the cumulative revenue generated from the customer and may have an impact on churn.
# 
# 21. Churn: The target variable indicates whether the customer has churned (canceled the service) or not. It is the main variable to predict in telecom customer churn analysis.
#  </div>

# -----
# ## Task 1: Problem definition
# -----
# <span style="font-size: larger;">Connectell is currently grappling with a customer churn issue, with initial analysis revealing that 25% of their customer base has already discontinued their services. This has unquestionably had a significant impact on Connectell's revenue. The main aim of this analysis is to harness business data and publicly accessible information to evaluate Connectell's offerings. The objective is to discern the factors contributing to customer attrition and employ machine learning algorithms to forecast potential churn based on existing data patterns. This examination will empower the company to pinpoint crucial predictor variables and undertake proactive measures to devise and execute effective strategies, with the ultimate goal of minimizing customer churn as much as possible.</span> 
# 
# <div style="line-height: 2;">
# 
# </div>
# 
# 

# ---
# ## Task 2 - DATA ANALYSIS 
# ---

# ### 2.1. Import Libraries
# ---

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings
from keras.models import Sequential    # The model class is Sequentially arranged 
from keras.layers import Dense, Dropout  # Dense indicates fully connected networks
from keras.losses import mean_squared_error  # loss 
from keras.optimizers import SGD # Evaluation metric
from keras.metrics import mean_squared_error         # Evaluation metric
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression


# ### 2.2. Loading data with Pandas
# ---

# In[2]:


df_churn = pd.read_csv("Customer-Churn - Customer-Churn.csv")


# In[3]:


df_churn.head() #Quick Glance on the data


# This glance through has revealed that the data comprises majorly of categorial data and fewer numeric data

# In[4]:


df_churn.shape


# - The dataset consists of `21` columns and `7043 rows`

# ### 2.3. Descriptive Statistics of the Churn Dataset

# #### Data Types 

# In[5]:


df_churn.dtypes


# From the types:
# - All data except "Senior Citizen", "tenure", "Monthly Charges", "Total Charges" are numerical data while others are categorical.
# - It is useful for understanding the nature of each feature in the dataset, as it provides information on whether a column contains numerical data (integers or floats) or categorical data (objects). It's a crucial step in the data analysis process to ensure appropriate data preprocessing and modeling techniques are applied based on the nature of the data.

# #### Dataset Description

# In[6]:


df_churn.describe()


# ##### - There are total counts of 7043 entries across the dataset
# - 25% of the customers has a tenure of 9 months with monthly charges and total charges valued at 35.5 and 401.45 respectively
# - 50% of the customers has a tenure of 29 months with monthly charges and total charges valued at 70.35 and 1397.48 respectively
# - 75% of the customers has a tenure of 55 months with monthly charges and total charges valued at 89.85 and 3794.74 respectively
# - The tenure's standard deviation, approximately 24.56, suggests a considerable spread in customer subscription durations. Moreover, the standard deviation of approximately 30.09 reflects variability in monthly charges, while the standard deviation of approximately 2266.77 indicates significant variability in total charges.
# - The average tenure period is 32.37 months
# - The average monthly charges across the distribution is 64.76 while the average total charges is 2283.30
# 

# In[7]:


df_churn.describe(include = 'object').T


# - Object data description showed that all data are unique based on the customer ID

# #### Checking for Missing Values (Null Data Analysis)
# ---

# In[8]:


df_churn.isnull().sum()


# It could be seen that there are 11 missing values in column 'Total Charges'; which calls for investigation.

# In[9]:


df_churn['TotalCharges'].unique()


# In[10]:


#Replacing the missing values with most common denominator as mode being the most occuring value within the column distribution
mode_value = df_churn['TotalCharges'].mode()
#Filling up
mode_value


# In[11]:


df_churn['TotalCharges'] = df_churn['TotalCharges'].fillna(20.2)


# In[12]:


df_churn.isnull().sum()  #to confirm the filling task done


# In[13]:


df_churn.info()


# In[14]:


df_churn['tenure'].unique()


# ---
# ### 2.4 EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ---

# In[15]:


sns.heatmap(df_churn.isnull(), yticklabels = False, annot = True)


# ###### ANALYSE RELATIONSHIP BETWEEN DIFFERENT FEATURES IN THE COLUMN AND THE CHURN

# In[16]:


# Set the style for the plots
sns.set(style="whitegrid")

# Analyze relationships between categorical features and churn
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Plot countplots for each categorical feature
plt.figure(figsize=(20, 20))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(4, 4, i)
    sns.countplot(x=feature, hue='Churn', data=df_churn)
    plt.title(f'{feature} vs Churn')

plt.tight_layout()
plt.show()



# In[17]:


# Analyze relationships between numerical features and churn
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges','SeniorCitizen']

# Plot boxplots for each numerical feature
plt.figure(figsize=(15, 5))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(1, 4, i)
    sns.boxplot(x='Churn', y=feature, data=df_churn)
    plt.title(f'{feature} vs Churn')

plt.tight_layout()
plt.show()


# In[18]:


# Create a violin plot
sns.violinplot(data=df_churn, x='Churn', y='tenure', hue='SeniorCitizen', split=True)

# Show the plot
plt.show()


# In[19]:


sns.scatterplot(data  = df_churn, x ='SeniorCitizen', y = 'tenure',hue ='Churn' )


# In[20]:


sns.displot(data=df_churn, x='gender', hue='Churn', binwidth=5)


# In[21]:


sns.scatterplot(data=df_churn, x='TotalCharges', y='tenure',hue ='Churn')


# In[22]:


sns.scatterplot(data=df_churn, x='MonthlyCharges', y='tenure', hue='SeniorCitizen')


# In[23]:


sns.scatterplot(data=df_churn, x='MonthlyCharges', y='tenure', hue='Churn')


# In[24]:


#checking the churn with the gender identity 'Male'.
Male_gender = df_churn.loc[df_churn['gender'] == 'Male']


# In[25]:


Male_gender.describe()


# In[26]:


#checking the churn with the gender identity 'Female'.
Female_gender = df_churn.loc[df_churn['gender'] == 'Female']


# In[27]:


Female_gender.describe()


# In[28]:


# Convert 'tenure' to integers since it contains objects
df_churn['tenure'] = pd.to_numeric(df_churn['tenure'], errors='coerce').astype('Int64')

# Define bins and labels
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
order_class = ['70-74','65-69','60-64','55-59','50-54','45-49','40-44','35-39','30-34','25-29','20-24','15-19','10-14','5-9','0-4']
labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74']

# Apply pd.cut after ensuring 'tenure' is numeric
df_churn['tenure Range'] = pd.cut(df_churn['tenure'], bins, labels=labels, include_lowest=True, ordered=False)


# In[29]:


df_churn['tenure Range']


# In[30]:


#this will group the data by tenure and gender
reg_data = df_churn.groupby(['tenure Range', 'gender']).size().reset_index(name='Churn')


# In[31]:


#This will create two differnt tables of male and female population, in order to perform an outer join later
female_table = reg_data [reg_data['gender'] == 'Female']
male_table = reg_data[reg_data['gender'] == 'Male']
female_table.columns = ['tenure Range', 'Gender', 'Female Churn']
male_table.columns = ['tenure Range', 'Gender', 'Male Churn']


# In[32]:


Pyramid_table = pd.merge(female_table, male_table, on='tenure Range', how='outer')
Pyramid_table


# In[33]:


pyramid_table =  Pyramid_table.drop(['Gender_x', 'Gender_y' ],axis = 1)


# In[34]:


pyramid_table['Churn'] = pyramid_table['Male Churn'] + pyramid_table['Female Churn']


# In[35]:


#This is required to create a similar plot for the age pyramid
pyramid_table['Male Churn'] = - (pyramid_table['Male Churn'])


# In[36]:


pyramid_table


# In[37]:


plt.figure(figsize=(12,8))
age_pyramid = sns.barplot(x='Male Churn', y='tenure Range', data=pyramid_table, color=('mediumblue'), label='Male', order= order_class )
age_pyramid = sns.barplot(x='Female Churn', y='tenure Range', data=pyramid_table,  color=('darkorange'), label='Female', order= order_class )
age_pyramid.set_title('Connectell Customer Churn Pyramid')
age_pyramid.set_xlabel('Churn (in hundreds)')
age_pyramid.set_ylabel('tenure Group')
age_pyramid.legend()


# In[38]:


pyramid_table= pyramid_table.sort_values('tenure Range', ascending= False, ignore_index = True)
pyramid_table


# In[39]:


# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(pyramid_table['tenure Range'], pyramid_table['Female Churn'], label='Female Churn', marker='o')
plt.plot(pyramid_table['tenure Range'], pyramid_table['Male Churn'], label='Male Churn', marker='o')
plt.plot(pyramid_table['tenure Range'], pyramid_table['Churn'], label='Churn', marker='o')

# Add labels and title
plt.xlabel('Tenure Range')
plt.ylabel('Number of Churns')
plt.title('Churn Curve by Gender and Total')
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# ### CHURN RATES

# In[40]:


# Calculate churn rate for Female and Male
pyramid_table['Female Churn Rate'] = (pyramid_table['Female Churn'] / pyramid_table['Churn']).abs() * 100
pyramid_table['Male Churn Rate'] = (pyramid_table['Male Churn'] / pyramid_table['Churn']).abs() * 100

# Display the DataFrame with churn rates
pyramid_table[['tenure Range', 'Female Churn Rate', 'Male Churn Rate']]


# In[41]:


df_churn.groupby(['gender','PaymentMethod'])['TotalCharges'].size().unstack().plot(kind='bar',stacked=True, title = "PaymentMethod_gender_Status")
plt.show()


# In[42]:


df_churn.groupby(['SeniorCitizen','InternetService'])['TotalCharges'].size().unstack().plot(kind='bar',stacked=True, title = "SeniorCitizen_InternetServices_Status")
plt.show()


# In[43]:


df_churn.groupby(['StreamingTV','InternetService'])['MonthlyCharges'].size().unstack().plot(kind='bar',stacked=True, title = "Monthly_Streaming_Charges")
plt.show()


# In[44]:


df_churn.groupby(['tenure Range','Churn'])['TotalCharges'].size().unstack().plot(kind='bar',stacked=True, title = "Tenure_Churn_Status")
plt.show()


# In[45]:


df_churn.groupby(['tenure Range','Churn'])['MonthlyCharges'].size().unstack().plot(kind='bar',stacked=True, title = "Tenure_Churn_Monthlycharges")
plt.show()


# In[46]:


# Melt the DataFrame to make it suitable for Seaborn's barplot
df_melted = pd.melt(pyramid_table, id_vars=['tenure Range', 'Churn'], value_vars=['Female Churn', 'Male Churn'], var_name='Gender', value_name='MonthlyCharges')

# Plot using Seaborn's barplot
plt.figure(figsize=(12, 8))
sns.barplot(x='tenure Range', y='MonthlyCharges', hue='Gender', data=df_melted, palette='Set2', ci=None)
plt.title("Tenure_Churn_Monthlycharges with Gender")
plt.show()


# In[47]:


sns.barplot(df_churn, x = 'tenure Range', y = 'TotalCharges')


# In[48]:


plt.figure(figsize= (16,8))

Religion_plot = sns.countplot(data = df_churn, x = 'tenure Range', hue = 'DeviceProtection'  )
Religion_plot.set_title('Tenure Range by DeviceProtection')
Religion_plot.set_ylabel('Tenure_DeviceProtection')


# In[49]:


plt.figure(figsize= (16,8))

Religion_plot = sns.countplot(data = df_churn, x = 'tenure Range', hue = 'Churn'  )
Religion_plot.set_title('Tenure Range by Churn')
Religion_plot.set_ylabel('Tenure_Churn')


# In[50]:


plt.figure(figsize=(16, 8))

Religion_plot = sns.countplot(data=df_churn, x='tenure Range', hue='gender', palette='Set2')
Religion_plot.set_title('Tenure Range by Churn and Gender')
Religion_plot.set_ylabel('Tenure_Churn')


# In[51]:


df_churn['PhoneService'].unique()


# In[52]:


# Create a second plot for 'phone services'
plt.figure(figsize=(16, 8))
Services_plot = sns.countplot(data=df_churn, x='tenure Range', hue='PhoneService', palette='Set2')
Services_plot.set_title('Tenure Range by Churn and Phone Services')
Services_plot.set_ylabel('Tenure_Churn')

# Add a legend for phone services
plt.legend(title='Phone Services', loc='upper right', labels=['Yes', 'No'])

plt.show()


# In[53]:


# Assuming 'tenure Range' is a column in pyramid_table
# Split 'tenure Range' into two columns
pyramid_table[['start', 'end']] = pyramid_table['tenure Range'].str.split('-', expand=True)

# Convert 'start' and 'end' to numeric
pyramid_table[['start', 'end']] = pyramid_table[['start', 'end']].apply(pd.to_numeric, errors='coerce')

# Calculate the average and create a new column 'average_tenure'
pyramid_table['average_tenure'] = (pyramid_table['start'] + pyramid_table['end']) / 2

# Plot the lmplot
plt.figure(figsize=(12, 8))
sns.lmplot(data=pyramid_table, x='average_tenure', y='Churn', hue='Male Churn Rate')
plt.title('Linear Regression Plot for Churn')
plt.show()


# In[54]:


# Assuming 'tenure Range' is a column in pyramid_table
# Split 'tenure Range' into two columns
pyramid_table[['start', 'end']] = pyramid_table['tenure Range'].str.split('-', expand=True)

# Convert 'start' and 'end' to numeric
pyramid_table[['start', 'end']] = pyramid_table[['start', 'end']].apply(pd.to_numeric, errors='coerce')

# Calculate the average and create a new column 'average_tenure'
pyramid_table['average_tenure'] = (pyramid_table['start'] + pyramid_table['end']) / 2

# Plot the lmplot
plt.figure(figsize=(12, 8))
sns.lmplot(data=pyramid_table, x='average_tenure', y='Churn', hue='Female Churn Rate')
plt.title('Linear Regression Plot for Churn')
plt.show()


# #### FURTHER ANALYSING THE CHURN BOTH YES AND NO

# In[55]:


#Considering when the Churn is Yes
churn_yes = df_churn[df_churn['Churn'] == 'Yes']
churn_yes


# In[56]:


plt.figure(figsize= (16,7))
churn_yes_plot = sns.countplot(data = churn_yes, x = 'tenure Range', hue = 'Partner' )
churn_yes_plot.set_title  ('Yes Churn Count by Partner')


# In[57]:


plt.figure(figsize= (16,7))
churn_yes_dependents_plot = sns.countplot(data = churn_yes, x = 'tenure Range', hue = 'Dependents' )
churn_yes_dependents_plot.set_title  ('Yes Churn Count by Dependents')


# In[58]:


plt.figure(figsize= (16,7))
churn_yes_multipleLine_plot = sns.countplot(data = churn_yes, x = 'tenure Range', hue = 'MultipleLines' )
churn_yes_multipleLine_plot.set_title('Yes Churn Count by MultipleLines')


# In[59]:


# Map 'tenure Range' to numerical values
churn_yes['tenure Range Numeric'] = churn_yes['tenure Range'].str.extract('(\d+)').astype(float)

# Plot lmplot
plt.figure(figsize=(10, 6))
sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='TotalCharges', hue='gender')
plt.show()


# In[60]:


sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='MonthlyCharges', hue='Contract')


# In[61]:


sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='TotalCharges', hue='Contract')


# In[62]:


sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='TotalCharges', hue='StreamingMovies')


# In[63]:


sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='TotalCharges', hue='OnlineBackup')


# In[64]:


sns.lmplot(data=churn_yes, x='tenure Range Numeric', y='TotalCharges', hue='OnlineSecurity')


# In[65]:


#Considering when the Churn is No
churn_no = df_churn[df_churn['Churn'] == 'No']
churn_no


# In[66]:


plt.figure(figsize= (16,7))
churn_no_plot = sns.countplot(data = churn_no, x = 'tenure Range', hue = 'Partner' )
churn_no_plot.set_title  ('No Churn Count by Partner')


# In[67]:


plt.figure(figsize= (16,7))
churn_no_dependents_plot = sns.countplot(data = churn_no, x = 'tenure Range', hue = 'Dependents' )
churn_no_dependents_plot.set_title  ('No Churn Count by Dependents')


# In[68]:


plt.figure(figsize= (16,7))
churn_no_multipleLine_plot = sns.countplot(data = churn_no, x = 'tenure Range', hue = 'MultipleLines' )
churn_no_multipleLine_plot.set_title('No Churn Count by MultipleLines')


# In[69]:


# Map 'tenure Range' to numerical values
churn_no['tenure Range Numeric'] = churn_no['tenure Range'].str.extract('(\d+)').astype(float)

# Plot lmplot
plt.figure(figsize=(10, 6))
sns.lmplot(data=churn_no, x='tenure Range Numeric', y='TotalCharges', hue='gender')
plt.show()


# In[70]:


sns.lmplot(data=churn_no, x='tenure Range Numeric', y='MonthlyCharges', hue='Contract')


# In[71]:


sns.lmplot(data=churn_no, x='tenure Range Numeric', y='TotalCharges', hue='Contract')


# In[72]:


sns.lmplot(data=churn_no, x='tenure Range Numeric', y='TotalCharges', hue='StreamingMovies')


# In[73]:


sns.lmplot(data=churn_no, x='tenure Range Numeric', y='TotalCharges', hue='OnlineBackup')


# In[74]:


sns.lmplot(data=churn_no, x='tenure Range Numeric', y='TotalCharges', hue='OnlineSecurity')


# ### TASK 3. PERFORMING PRINCIPAL COMPONENT ANALYSIS ON THE DATAFRAME
# When dealing with a large number of features, PCA can help in reducing the dimensionality of the dataset. This is particularly useful when there are correlated features. This can be useful for visual exploration of patterns and potential clusters related to churn.

# In[75]:


#encode churn column into numeral values using LabelEncoder
le = LabelEncoder()

Churn_class = le.fit_transform(df_churn['Churn'])
df_churn['Churn'] = Churn_class
Churn_class


# In[76]:


# Assuming 'churn' is the original DataFrame
cols = ['SeniorCitizen', 'tenure Range', 'MonthlyCharges', 'TotalCharges', 'Churn']

X_data = df_churn[cols].copy() # Use .copy() to create a new DataFrame and avoid modifying the original DataFrame

# Display the new DataFrame
X_data.head()


# In[77]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Identify numeric and categorical features
numeric_features = X_data.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X_data.select_dtypes(include=['object']).columns

# Create transformers for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[78]:


# Apply the preprocessing steps
X_preprocessed = preprocessor.fit_transform(X_data)


# In[79]:


scaler = StandardScaler()

pca_churn_std = scaler.fit_transform(X_preprocessed)

pca_churn_std_df = pd.DataFrame(pca_churn_std)
pca_churn_std_df


# In[80]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[81]:


pca=PCA().fit(pca_churn_std_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance')
plt.show()


# In[82]:


pca=PCA(n_components=2)
new_data= pca.fit_transform(pca_churn_std_df)
new_df = pd.DataFrame(new_data, columns=['PC1','PC2'])
new_df


# In[83]:


new_df['Churn'] = X_data['Churn']
new_df


# In[84]:


#Let's also visualize the outcome of the PCA.

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
targets = X_data['Churn']
colors = ['r', 'g', 'c']
for target, color in zip(targets,colors):
    indicesToKeep = new_df['Churn'] == target
    ax.scatter(new_df.loc[indicesToKeep, 'PC1']
               , new_df.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[85]:


X = np.array(X_data['Churn']).reshape(-1, 1)


# In[86]:


ss_distance = []
K = range(1, 10)
for clusters in K:
    K_means_clusters = KMeans(n_clusters = clusters, n_init=3, random_state = 25)
    K_means_clusters.fit(X)
    ss_distance.append(K_means_clusters.inertia_)

plt.title('Sum of Square Distance vs Number of Clusters')   
plt.plot(K, ss_distance)
plt.xlabel('Number of Clusters' ,fontsize='12')
plt.ylabel('Distance Sum of Square', fontsize='12')
plt.show()


# In[87]:


kmeans = KMeans(n_clusters=3, random_state=0)
X_data['labels'] = kmeans.fit_predict(X)
X_data


# In[88]:


plt.figure(figsize=(4,4),dpi=100)
sns.scatterplot(data=X_data, x=X_data['Churn'],y=X_data['Churn'],hue=X_data['labels'] ,palette='tab10')
plt.show()


# In[89]:


from sklearn.metrics import silhouette_score, davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)
silhouette_avg = silhouette_score(X, cluster_labels)
dbs = davies_bouldin_score(X, cluster_labels)

print('Silhouette score =', silhouette_avg)
print('Davies Bouldin score =', dbs)


# In[90]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import ClusterNode


# In[91]:


matrix_linkage = hierarchy.linkage(kmeans.cluster_centers_) # hierarchy is used to obtain the clusters
matrix_linkage # distances between the points 


# In[92]:


plt.figure(figsize=(20,12), dpi = 200,)
dendrogram(Z=matrix_linkage,truncate_mode= 'level', p= 7);


# ### TASK 4 SUPERVISED MACHINE LEARNING

# In[93]:


numeric = df_churn.select_dtypes(exclude = 'object')
objects = df_churn.select_dtypes(include = 'object')


# In[94]:


numeric= numeric.drop('tenure Range', axis=1)
numeric.head()


# In[95]:


objects.head()


# In[96]:


df_churn['Churn'].unique()


# In[97]:


sns.countplot(data=df_churn, x='Churn')


# In[98]:


sns.heatmap(numeric.corr(), annot=True)


# In[99]:


# Apply LabelEncoder to each object column
encoded_objects = pd.DataFrame()
for column in objects.columns:
    enc = LabelEncoder()
    encoded_objects[column] = enc.fit_transform(objects[column])

# Combine the numeric and encoded object DataFrames
combined_df = pd.concat([numeric, encoded_objects], axis=1)


# In[100]:


sns.heatmap(combined_df.corr(), annot=True)
plt.show()


# In[101]:


# Assuming 'Churn' is the column you want to analyze
target_column = 'Churn'

# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Extract correlations with the target column
correlations_with_target = correlation_matrix[target_column].drop(target_column)

# Identify positively and negatively correlated variables
positively_correlated = correlations_with_target[correlations_with_target > 0].index
negatively_correlated = correlations_with_target[correlations_with_target < 0].index


# In[102]:


# Print positively and negatively correlated variables
print("Positively correlated variables:")
print(positively_correlated)


# In[103]:


print("\nNegatively correlated variables:")
print(negatively_correlated)


# In[104]:


combined_df.boxplot(figsize = (15,10), fontsize = 15, rot = 60)


# ## Building the Model

# In[105]:


from sklearn.model_selection import train_test_split

X = combined_df.drop('Churn',axis=1)
y = combined_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = X_train
y= y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)


# ## Normalizing the Dataset

# In[106]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_Xtrain = scaler.fit_transform(X_train)
scaled_Xtest = scaler.transform(X_test)
scaled_Xval = scaler.transform(X_val)
Xtrain_df = pd.DataFrame(scaled_Xtrain, columns = X_train.columns)
Xtest_df = pd.DataFrame(scaled_Xtest, columns = X_test.columns)
Xval_df = pd.DataFrame(scaled_Xval, columns = X_val.columns)


# ## Simple Linear Regression

# In[107]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
cont_variables = Xtrain_df[['SeniorCitizen', 'MonthlyCharges', 'PhoneService', 'MultipleLines',
       'PaperlessBilling', 'PaymentMethod']]
for col in cont_variables.columns:
    lm = LinearRegression()
    col_xtrain = Xtrain_df[col].to_numpy().reshape(-1,1)
    col_xval = Xval_df[col].to_numpy().reshape(-1,1)
    lm.fit(col_xtrain, y_train)
    y_pred = lm.predict(col_xval)
    r2 = r2_score(y_val, y_pred)
    mean_abs_err = mean_absolute_error(y_val, y_pred)
    mean_sqr_err = mean_squared_error(y_val, y_pred)
    root_mean_sqr = np.sqrt(mean_sqr_err)
    print(f"{col}")
    print(f"Mean Absolute Error: {mean_abs_err:.2f}")
    print(f"Root Mean Square Error: {root_mean_sqr:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print('\n')


# ## USING ALL VARIABLES TO PREDICT CHURN

# In[108]:


X = combined_df.drop('Churn', axis=1)
y = combined_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X = X_train
y= y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)


# In[109]:


scaler = MinMaxScaler()
scaled_Xtrain = scaler.fit_transform(X_train)
scaled_Xtest = scaler.transform(X_test)
scaled_Xval = scaler.transform(X_val)
Xtrain_df = pd.DataFrame(scaled_Xtrain, columns = X_train.columns)
Xtest_df = pd.DataFrame(scaled_Xtest, columns = X_test.columns)
Xval_df = pd.DataFrame(scaled_Xval, columns = X_val.columns)


# In[110]:


lm = LinearRegression()
lm.fit(Xtrain_df, y_train)
y_pred = lm.predict(Xval_df)
r2 = r2_score(y_val, y_pred)
mean_abs_err = mean_absolute_error(y_val, y_pred)
mean_sqr_err = mean_squared_error(y_val, y_pred)
root_mean_sqr = np.sqrt(mean_sqr_err)


# In[111]:


print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mean_abs_err:.2f}")
print(f"Root Mean Square Error: {root_mean_sqr:.2f}")


# #### For the Negative Correlation

# In[112]:


corr = combined_df[['Churn','tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract']].corr()
corr


# ##### For the positive Correlations

# In[113]:


corre = combined_df[['Churn','SeniorCitizen', 'MonthlyCharges', 'PhoneService', 'MultipleLines',
       'PaperlessBilling', 'PaymentMethod']].corr()
corre


# In[114]:


#FOR THE VARIABLES WITH NEGATIVE CORRELATION
sub_Xtrain = Xtrain_df[['tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract']]
sub_Xtest = Xtest_df[['tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract']]
sub_Xval = Xval_df[['tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract']]

lm = LinearRegression()
lm.fit(sub_Xtrain, y_train)
y_pred = lm.predict(sub_Xval)

r2 = r2_score(y_val, y_pred)
mean_abs_err = mean_absolute_error(y_val, y_pred)
mean_sqr_err = mean_squared_error(y_val, y_pred)
root_mean_sqr = np.sqrt(mean_sqr_err)

print(f"Results when 'Partner', 'Dependents','OnlineSecurity',\n"
      f"'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',\n"
      f"'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'\n"
      f"are considered as categorical variables:\n")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mean_abs_err:.2f}")
print(f"Root Mean Square Error: {root_mean_sqr:.2f}")


# In[115]:


sub_Xtrain = Xtrain_df[['SeniorCitizen','MonthlyCharges','PaperlessBilling']]
sub_Xtest = Xtest_df[['SeniorCitizen','MonthlyCharges','PaperlessBilling']]
sub_Xval = Xval_df[['SeniorCitizen','MonthlyCharges','PaperlessBilling']]

lm = LinearRegression()
lm.fit(sub_Xtrain, y_train)
y_pred = lm.predict(sub_Xval)

r2 = r2_score(y_val, y_pred)
mean_abs_err = mean_absolute_error(y_val, y_pred)
mean_sqr_err = mean_squared_error(y_val, y_pred)
root_mean_sqr = np.sqrt(mean_sqr_err)

print(f"Results when 'SeniorCitizen','MonthlyCharges' were used:")
print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mean_abs_err:.2f}")
print(f"Root Mean Square Error: {root_mean_sqr:.2f}")


# ## Testing

# In[116]:


lm = LinearRegression()
lm.fit(sub_Xtrain, y_train)
y_pred = lm.predict(sub_Xtest)

r2 = r2_score(y_test, y_pred)
mean_abs_err = mean_absolute_error(y_test, y_pred)
mean_sqr_err = mean_squared_error(y_test, y_pred)
root_mean_sqr = np.sqrt(mean_sqr_err)

print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mean_abs_err:.2f}")
print(f"Root Mean Square Error: {root_mean_sqr:.2f}")


# In[117]:


plt.title('CUSTOMER CHURN ANALYSIS')
plt.xticks(rotation=60)
sns.lineplot(data=combined_df, x='MonthlyCharges', y='Churn')


# ### FURTHER CLASSIFICATION

# In[118]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
 confusion_matrix, ConfusionMatrixDisplay, roc_auc_score)


# In[119]:


X = combined_df.drop(columns=['tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract','Churn'])
                      
y = combined_df['Churn']


# In[120]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=21, k_neighbors = 2) 
X_sm, y_sm = sm.fit_resample(X, y) 
X, y = X_sm, y_sm


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21) 
X,y=X_train,y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)


# In[122]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


# In[123]:


models = [('Decision Tree', DecisionTreeClassifier()), ('Support Vector', SVC()),
         ('Random Forest', RandomForestClassifier()),('Logistic Regression', LogisticRegression()), 
          ('K Neighbors', KNeighborsClassifier())]
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    print(name)
    print(classification_report(y_val,y_pred))
    ConfusionMatrixDisplay.from_estimator(model, X_val_scaled, y_val)
    plt.show()


# ###### Using Cross Validation to ensure there was no overfitting

# In[124]:


from sklearn.model_selection import cross_val_score, KFold

for name, model in models:
    kfold = KFold(n_splits = 7)
    cross_val = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f'{name}: {cross_val.mean():.2f} mean accuracy, {cross_val.std():.2f} standard deviation')   


# In[ ]:





# In[125]:


X = combined_df.drop(columns=['tenure', 'TotalCharges', 'customerID', 'gender', 'Partner',
       'Dependents', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract','Churn','PhoneService', 'MultipleLines'])
                      
y = combined_df['Churn']


# In[126]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=21, k_neighbors = 2) 
X_sm, y_sm = sm.fit_resample(X, y) 
X, y = X_sm, y_sm


# In[127]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21) 
X,y=X_train,y_train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=21)


# In[128]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


# In[129]:


models = [('Decision Tree', DecisionTreeClassifier()), ('Support Vector', SVC()),
         ('Random Forest', RandomForestClassifier()),('Logistic Regression', LogisticRegression()), 
          ('K Neighbors', KNeighborsClassifier())]
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    print(name)
    print(classification_report(y_val,y_pred))
    ConfusionMatrixDisplay.from_estimator(model, X_val_scaled, y_val)
    plt.show()


# #### CROSS VALIDATION

# In[130]:


from sklearn.model_selection import cross_val_score, KFold

for name, model in models:
    kfold = KFold(n_splits = 5)
    cross_val = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f'{name}: {cross_val.mean():.2f} mean accuracy, {cross_val.std():.2f} standard deviation') 


# In[ ]:




