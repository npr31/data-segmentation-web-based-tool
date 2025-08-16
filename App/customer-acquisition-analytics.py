#!/usr/bin/env python
# coding: utf-8

# #### Problem Statement 
# The data is related to a marketing campaign done by a Portuguese Bank. The campaigns were telephonic, i.e., sales agents made phone calls to sell a term deposit product. 
# 
# The business objectives are:-
# - To reduce customer acquisition cost by targeting the ones who are likely to buy
# - To improve the response rate, i.e., the fraction of prospects who respond to the campaign

# We will follow below steps:-
# 
# - Read and understand the data
# - Exploratort data analysis
# - Prepare the data for modelling
# - Modle evaluation
# - Create Gain and Lift charts, and finacial benefits for the banks for customer acquisition by using the model
import sys
import pandas as pd

def process_customer_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Example Processing: Filtering high-value customers
    df["Customer_Value"] = df["Total_Purchase"] / df["Num_Transactions"]
    high_value_customers = df[df["Customer_Value"] > 1000]

    # Save processed data
    high_value_customers.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_customer_data(input_file, output_file)

# In[101]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns',100)


# In[102]:


import warnings
warnings.filterwarnings('ignore')


# ## Read and understand the data

# In[103]:


data = pd.read_csv('bank_marketing.csv')
data.head()


# In[104]:


data.shape


# In[105]:


data.info()


# In[106]:


data.describe()


# #### Drop duration column
# Because, when an executive picks up the phone and calls a prospective client, he/she is unsure of the duration of the call. 

# In[107]:


# Drop duration
data.drop('duration', axis=1, inplace=True)


# #### Treating Missing Values in columns

# In[ ]:


# Cheking percent of missing values in columns
(data.isnull().sum()/len(data.index))*100


# We can see there is no missing values in any column.

# ### Checking response variable

# In[109]:


data['response'].value_counts()


# We can see that there is an imbalance in the response rate.

# In[110]:


# Converting yes to 1 an and no to 0
data['response'] = data['response'].map({'yes':1, 'no':0})


# In[111]:


data['response'].head()


# In[112]:


# Response rate
round((data['response'].sum()/len(data.index))*100,2)


# ## Exploratory Data Analysis

# First, we will look at the client data.
# 
# - Age
# - Job : type of job
# - Marital : marital status
# - Education
# - Default: has credit in default?
# - Housing: has housing loan?
# - Loan: has personal loan?

# #### Age

# In[113]:


sns.boxplot(x='age', data=data)


# We can see above 70 years of age there are many outliers. We can cap the age to 70 so that the model behaviour doesn't change significantly with the age above 70.

# In[114]:


data['age'][data['age']>70] = 70


# In[115]:


sns.boxplot(x='age', data=data)


# In[116]:


# create age bins
bins = np.arange(10,data['age'].max()+10,10)
data['age_bins'] = pd.cut(data['age'], bins)
data['age_bins']


# In[117]:


no_of_prospects = data.groupby('age_bins')['response'].count().reset_index(name='no_of_prospects')
no_of_prospects


# In[118]:


count_yes_prospects = data[data['response']==1].groupby('age_bins')['response'].count().reset_index(name='count_yes_prospects')
count_yes_prospects


# In[119]:


age_df = no_of_prospects.merge(count_yes_prospects, on='age_bins')
age_df.head()


# In[120]:


age_df['response_rate'] = ((age_df['count_yes_prospects']/age_df['no_of_prospects'])*100).round(2)
age_df


# In[121]:


sns.barplot(x='age_bins', y='response_rate', data=age_df)
plt.show()


# We can see that the youngest and eldest have more response rate than others, keeping in mind that number of prospects is very less for the youngest.

# #### Job

# In[122]:


data['job'].value_counts()


# In[123]:


no_of_prospects = data.groupby('job')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('job')['response'].sum().reset_index(name='count_yes_prospects')
job_df = no_of_prospects.merge(count_yes_prospects, on='job')
job_df['response_rate'] = ((job_df['count_yes_prospects']/job_df['no_of_prospects'])*100).round(2)
job_df


# In[124]:


plt.figure(figsize=(15,7))
sns.barplot(x='job', y='response_rate', data=job_df)
plt.show()


# We can draw similarities from the age response analysis where we found that the youngest and eldest were most likely to respond in a positive manner. It is rreiterated by the above analysis, where we notice that student and retired have the highest response rates.

# #### Marital

# In[125]:


data['marital'].value_counts()


# In[126]:


no_of_prospects = data.groupby('marital')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('marital')['response'].sum().reset_index(name='count_yes_prospects')
marital_df = no_of_prospects.merge(count_yes_prospects, on='marital')
marital_df['response_rate'] = ((marital_df['count_yes_prospects']/marital_df['no_of_prospects'])*100).round(2)
marital_df


# In[127]:


sns.barplot(x='marital', y='response_rate', data=marital_df)
plt.show()


# We can see that singles are more likely to response positive that other groups.

# #### Education

# In[128]:


data['education'].value_counts()


# To simplify the problem, we are going to convert the educational qualifications to simple buckets of primary educations, secondary education, tertiary education and professional courses. Also note that there are 18 entries listing the educational status as illiterate. Since the closest group to them is primary education, we are going to assign all entries with educational status as illiterate to primary education.

# In[129]:


data.replace({'education' : { 'basic.9y' : 'Primary_Education', 'basic.4y' : 'Primary_Education','basic.6y' : 'Primary_Education','illiterate' : 'Primary_Education',
                             'high.school' : 'Secondary_Education', 'university.degree' : 'Tertiary_Education' }}, inplace=True)


# In[130]:


data['education'].value_counts()


# In[131]:


no_of_prospects = data.groupby('education')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('education')['response'].sum().reset_index(name='count_yes_prospects')
education_df = no_of_prospects.merge(count_yes_prospects, on='education')
education_df['response_rate'] = ((education_df['count_yes_prospects']/education_df['no_of_prospects'])*100).round(2)
education_df


# In[132]:


plt.figure(figsize=(15,7))
sns.barplot(x='education', y='response_rate', data=education_df)
plt.show()


# We van see that Primary Education group has the lowest response rate, which is reasonable because may be because of lack of education people are not able to trust the bank.

# #### Previous Default

# In[133]:


# Default column
data['default'].value_counts()


# As we can see that most of the values are unknown and only three values are yes for default column. Hence, it is better to drop the column as the colimn doesn't have much variance.

# In[134]:


#Dropping default column
data.drop('default', axis=1, inplace=True)


# #### Housing
# 

# In[135]:


data['housing'].value_counts()


# In[136]:


no_of_prospects = data.groupby('housing')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('housing')['response'].sum().reset_index(name='count_yes_prospects')
housing_df = no_of_prospects.merge(count_yes_prospects, on='housing')
housing_df['response_rate'] = ((housing_df['count_yes_prospects']/housing_df['no_of_prospects'])*100).round(2)
housing_df


# In[137]:


sns.barplot(x='housing', y='response_rate', data=housing_df)
plt.show()


# We can see that there is a slight uptick on the response rate for the people having housing.

# #### Loan

# In[138]:


data['loan'].value_counts()


# In[139]:


no_of_prospects = data.groupby('loan')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('loan')['response'].sum().reset_index(name='count_yes_prospects')
loan_df = no_of_prospects.merge(count_yes_prospects, on='loan')
loan_df['response_rate'] = ((loan_df['count_yes_prospects']/loan_df['no_of_prospects'])*100).round(2)
loan_df


# In[140]:


sns.barplot(x='loan', y='response_rate', data=loan_df)
plt.show()


# There in no notable difference between the groups having housing loan on the response rate of the investment product.

# Now we will look at the client data.
# - Contact: contact communication type
# - Month: last contact month of year
# - Day_of_week: last contact day of the week

# #### Contact

# In[141]:


data['contact'].value_counts()


# In[142]:


no_of_prospects = data.groupby('contact')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('contact')['response'].sum().reset_index(name='count_yes_prospects')
contact_df = no_of_prospects.merge(count_yes_prospects, on='contact')
contact_df['response_rate'] = ((contact_df['count_yes_prospects']/contact_df['no_of_prospects'])*100).round(2)
contact_df


# In[143]:


sns.barplot(x='contact', y='response_rate', data=contact_df)
plt.show()


# Here the difference is clear. The prospects lcontacted with cellular medimum are more likely to convert as a customer.

# #### Month

# In[144]:


data['month'].value_counts()


# In[145]:


no_of_prospects = data.groupby('month')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('month')['response'].sum().reset_index(name='count_yes_prospects')
month_df = no_of_prospects.merge(count_yes_prospects, on='month')
month_df['response_rate'] = ((month_df['count_yes_prospects']/month_df['no_of_prospects'])*100).round(2)
month_df


# In[146]:


sns.barplot(x='month', y='response_rate', data=month_df)
plt.show()


# We can note that the four months of december, march, october and september appear to be the best to contact the potential customers. However, please note that these our months have the fewest data entries as well, so it is not certain, how well it would behave when calls are made at a high volume.

# #### Day of the week

# In[147]:


data['day_of_week'].value_counts()


# In[148]:


no_of_prospects = data.groupby('day_of_week')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('day_of_week')['response'].sum().reset_index(name='count_yes_prospects')
day_of_week_df = no_of_prospects.merge(count_yes_prospects, on='day_of_week')
day_of_week_df['response_rate'] = ((day_of_week_df['count_yes_prospects']/day_of_week_df['no_of_prospects'])*100).round(2)
day_of_week_df


# In[149]:


sns.barplot(x='day_of_week', y='response_rate', data=day_of_week_df)
plt.show()


# The response rate is highest on Thursday and lowest on Monday, the the difference is not much.

# Now, we will lok at the data related to last contacts.
# 
# - Campaign: number of contacts performed during this campaign and for this client
# - Pdays: number of days that passed by after the client was last contacted from a previous campaign
# - Previous: number of contacts performed before this campaign and for this client
# - Poutcome: outcome of the previous marketing campaign

# #### Campaign

# In[150]:


sns.distplot(data['campaign'])
plt.show()


# In[151]:


# Creating Box plot 
sns.boxplot(x='campaign', data=data)


# Looks like most of the campaign happened within the range of 10. We will cap all the contacts more than 10 to 10.

# In[152]:


data['campaign'][data['campaign']>10] = 10


# In[153]:


no_of_prospects = data.groupby('campaign')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('campaign')['response'].sum().reset_index(name='count_yes_prospects')
campaign_df = no_of_prospects.merge(count_yes_prospects, on='campaign')
campaign_df['response_rate'] = ((campaign_df['count_yes_prospects']/campaign_df['no_of_prospects'])*100).round(2)
campaign_df


# In[154]:


sns.barplot(x='campaign', y='response_rate', data=campaign_df)
plt.show()


# We can see that the response rate is higher, when the number of campaigns are becoming lower.

# #### pdays:-Gap after the last contact

# In[155]:


data['pdays'].value_counts()


# The value 999 means the prospects were first time contacted. 
# 
# We will bucket the coulm in three categories below.

# In[156]:


data['pdays'][data['pdays'].isin([0,1,2,3,4,5,6,7,8,9,10])] = 'contacted_in_first_10_days'


# In[157]:


data['pdays'][data['pdays']==999] = 'contacted_first_time'


# In[158]:


data['pdays'][~data['pdays'].isin(['contacted_in_first_10_days', 'contacted_first_time'])] = 'contacted_after_10_days'


# In[159]:


data['pdays'].value_counts()


# In[160]:


no_of_prospects = data.groupby('pdays')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('pdays')['response'].sum().reset_index(name='count_yes_prospects')
pdays_df = no_of_prospects.merge(count_yes_prospects, on='pdays')
pdays_df['response_rate'] = ((pdays_df['count_yes_prospects']/pdays_df['no_of_prospects'])*100).round(2)
pdays_df


# In[161]:


plt.figure(figsize=(10,5))
sns.barplot(x='pdays', y='response_rate', data=pdays_df)
plt.show()


# We can see that the people contacted within first 10 days are more likely to be converted.

# #### Previous

# In[162]:


data['previous'].value_counts()


# Since the data ranges from 0 to 7, we are going to break it into three categories.

# In[163]:


data['previous'][data['previous'].isin([0])]='never_contacted'
data['previous'][data['previous'].isin([1,2,3])]='less_than_3_times'
data['previous'][data['previous'].isin([4,5,6,7])]='more than_3_times'


# In[164]:


no_of_prospects = data.groupby('previous')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('previous')['response'].sum().reset_index(name='count_yes_prospects')
previous_df = no_of_prospects.merge(count_yes_prospects, on='previous')
previous_df['response_rate'] = ((previous_df['count_yes_prospects']/previous_df['no_of_prospects'])*100).round(2)
previous_df


# In[165]:


sns.barplot(x='previous', y='response_rate', data=previous_df)
plt.show()


# We can see that that the customers, who were contacted more than three time are very much likely to take the product.

# Now, we will look at the socio-economic variables.
# 
# - emp.var.rate: Employment Variation Rate - Quarterly Indicator
# - cons.price.idx: Consumer Price Index - Monthly Indicator 
# - cons.conf.idx: Consumer Confidence Index - Monthly Indicator 
# - euribor3m: Euribor 3 Month Rate - Daily Indicator
# - nr.employed: Number of Employees - Quarterly Indicator

# #### Employment Variation Rate

# In[166]:


sns.distplot(data['emp.var.rate'])
plt.show()


# In[167]:


no_of_prospects = data.groupby('emp.var.rate')['response'].count().reset_index(name='no_of_prospects')
count_yes_prospects = data[data['response']==1].groupby('emp.var.rate')['response'].sum().reset_index(name='count_yes_prospects')
emp_var_rate_df = no_of_prospects.merge(count_yes_prospects, on='emp.var.rate')
emp_var_rate_df['response_rate'] = ((emp_var_rate_df['count_yes_prospects']/emp_var_rate_df['no_of_prospects'])*100).round(2)
emp_var_rate_df


# In[168]:


sns.barplot(x='emp.var.rate', y='response_rate', data=emp_var_rate_df)
plt.show()


# We can note that a negative employment variation rate seems to be related to higher response rates.

# #### Consumer Price Index

# In[169]:


sns.distplot(data['cons.price.idx'])
plt.show()


# In[170]:


data['cons.price.idx'].describe(percentiles = [0.25,0.50,0.75,0.95,0.99])


# #### Number of employeees

# In[171]:


sns.distplot(data['nr.employed'])
plt.show()


# In[172]:


data['nr.employed'].describe(percentiles = [0.25,0.50,0.75,0.95,0.99])


# In[173]:


data.head()


# ## Data preparation for model building

# In[174]:


# Droping the column age_bins we had created for the purpose of data visualisation
data.drop(columns = 'age_bins', inplace = True)


# In[175]:


# Create dummy variables for job
job = pd.get_dummies(data['job'], prefix='job', drop_first=True)
#Concat data and job
data = pd.concat([data, job], axis=1)


# In[176]:


# Create dummy variables for marital
marital = pd.get_dummies(data['marital'], prefix='marital', drop_first=True)
#Concat data and marital
data = pd.concat([data, marital], axis=1)


# In[177]:


# Create dummy variables for education
education = pd.get_dummies(data['education'], prefix='education', drop_first=True)
#Concat data and education
data = pd.concat([data, education], axis=1)


# In[178]:


# Create dummy variables for housing
housing = pd.get_dummies(data['housing'], prefix='housing', drop_first=True)
#Concat data and housing
data = pd.concat([data, housing], axis=1)


# In[179]:


# Create dummy variables for loan
loan = pd.get_dummies(data['loan'], prefix='loan', drop_first=True)
#Concat data and loan
data = pd.concat([data, loan], axis=1)


# In[180]:


# Create dummy variables for contact
contact = pd.get_dummies(data['contact'], prefix='contact', drop_first=True)
#Concat data and contact
data = pd.concat([data, contact], axis=1)


# In[181]:


# Create dummy variables for month
month = pd.get_dummies(data['month'], prefix='month', drop_first=True)
#Concat data and month
data = pd.concat([data, month], axis=1)


# In[182]:


# Create dummy variables for day_of_week
day_of_week = pd.get_dummies(data['day_of_week'], prefix='day_of_week', drop_first=True)
#Concat data and day_of_week
data = pd.concat([data, day_of_week], axis=1)


# In[183]:


# Create dummy variables for pdays
pdays = pd.get_dummies(data['pdays'], prefix='pdays', drop_first=True)
#Concat data and pdays
data = pd.concat([data, pdays], axis=1)


# In[184]:


# Create dummy variables for poutcome
poutcome = pd.get_dummies(data['poutcome'], prefix='poutcome', drop_first=True)
#Concat data and poutcome
data = pd.concat([data, poutcome], axis=1)


# In[185]:


# Create dummy variables for previous
previous = pd.get_dummies(data['previous'],prefix='previous',drop_first=True)
data = pd.concat([data,previous],axis=1)


# In[186]:


# Drop all the columns for which dummy variables were created
data.drop(['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'previous','pdays'], axis=1, inplace=True)


# In[187]:


data.shape


# In[188]:


data.head()


# ### Train-Test split

# In[189]:


from sklearn.model_selection import train_test_split
# Putting feature variables to X
X = data.drop('response', axis=1)
#Putting target variable to y
y = data['response']


# In[190]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)


# ### Feature scaling

# In[191]:


# Numeric columns
num_cols = ['age','campaign','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']


# In[192]:


# Scaling train set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_train.head()


# In[193]:


# Scaling test set
# We will only transform the test set on basis of what was learnt from the train set
X_test[num_cols] = scaler.transform(X_test[num_cols])
X_test.head()


# ### Model building

# #### Principal Component Analysis

# In[194]:


from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X_train)


# In[195]:


# Cumuliative variance
pd.Series((pca.explained_variance_ratio_.cumsum()*100).round(2))


# In[196]:


# Scree Plot for cumuliative variance 
plt.plot((pca.explained_variance_ratio_.cumsum()*100).round(2))
plt.show()


# We can see that arount 15 variables explain 90% variance in the data.

# In[197]:


# PCA with 15 components
pca_best = PCA(n_components=15, random_state=42)
X_train_pca = pca_best.fit_transform(X_train)


# In[198]:


# Logistic model 
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(class_weight='balanced')


# In[199]:


# Fit the model
logistic_model.fit(X_train_pca, y_train)


# In[200]:


logistic_model.score(X_train_pca, y_train)


# #### Model evaluation on the train data

# In[201]:


# Prediction on the test set
y_train_pred = logistic_model.predict(X_train_pca)


# In[202]:


# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
cm = confusion_matrix(y_train, y_train_pred)
print(cm)


# In[203]:


# Sensitivity and specificity
from imblearn.metrics import sensitivity_specificity_support # type: ignore
sensitivity, specificity, _ = sensitivity_specificity_support(y_train, y_train_pred)
print("Sensitivity: ", round(sensitivity[0], 2), "\n", "Specificity: ", round(specificity[0], 2), sep='')


# In[204]:


# ROC-AUC Score
y_train_pred_prob = logistic_model.predict_proba(X_train_pca)[:, 1]
print("AUC: ", round(roc_auc_score(y_train, y_train_pred_prob),2))


# #### Model evaluation on the test data

# In[205]:


# Tranforming the test set in PCA
X_test_pca = pca_best.transform(X_test)


# In[206]:


logistic_model.score(X_test_pca, y_test)


# In[207]:


# Prediction on the test set
y_test_pred = logistic_model.predict(X_test_pca)


# In[208]:


# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)


# In[209]:


# Sensitivity and specificity
from imblearn.metrics import sensitivity_specificity_support
sensitivity, specificity, _ = sensitivity_specificity_support(y_test, y_test_pred)
print("Sensitivity: ", round(sensitivity[0], 2), "\n", "Specificity: ", round(specificity[0], 2), sep='')


# In[210]:


# check area under curve
y_test_pred_prob = logistic_model.predict_proba(X_test_pca)[:, 1]
print("AUC: ", round(roc_auc_score(y_test, y_test_pred_prob),2))


# ## Gain Chart and Lift Chart
# 
# - ***Gain chart:*** It plots the responders captured against the number of prospects targeted.
# 
#     -It tells us the number of responders captured (y-axis) as a function of the number of prospects targeted (x-axis).
#     
# 
# - ***Lift chart:*** It compares the response rate with and without using the model.
# 
#     -Compares the ‘lift in response rate’ you will get using the model viz-à-viz when you target the entire population (without      using the model)
# 
#     -Contains lift (on y-axis) and the number of prospects targeted (on x-axis)

# In[211]:


# Data frame for actual, predicted and predicted probabilities
df_proba = pd.DataFrame()
df_proba['actual'] = y_test
df_proba['pred_proba'] = y_test_pred_prob
df_proba['predicted'] = y_test_pred
df_proba.head()


# In[212]:


df_decile = df_proba


# In[213]:


# Creating deciles
df_decile['decile'] = pd.qcut(df_decile['pred_proba'], 10, np.arange(10,0,-1))
df_decile


# In[214]:


# Group by the total prospects on basis of the deciles
df_lift = df_decile.groupby('decile')['pred_proba'].count().reset_index()
df_lift.rename({'pred_proba':'total'}, axis=1, inplace=True)
df_lift


# In[215]:


# Creating dataframe for actual responses grouping by the deciles
df_lift_pred = df_decile[df_decile['actual']==1].groupby('decile')['actual'].count().reset_index()
df_lift_pred


# In[216]:


# Merging two dfs
df_lift_final = df_lift.merge(df_lift_pred, on='decile')
df_lift_final


# In[217]:


# Sort the deciles
df_lift_final = df_lift_final.sort_values('decile', ascending=False)
# Cumuliative response
df_lift_final['cum_response'] = df_lift_final['actual'].cumsum()
df_lift_final


# In[218]:


# Gain
df_lift_final['gain'] = round(100*(df_lift_final['cum_response']/df_lift_final['total']),2)
# Lift
df_lift_final['cum_lift'] = round(df_lift_final['gain']/(df_lift_final['decile'].astype('int')*10),2)
df_lift_final


# ### Gain Chart

# In[219]:


df_lift_final.plot(x='decile', y='gain')
plt.show()


# We can see from the above Gain chart that in 5th decile the conversion rate is almost 90%. That means, if we target only top 50% customers, we will be able to capture 90% responders.
# 
# Hence, we can acquire 90% customers with the 50% cost.

# ### Lift Chart

# In[220]:


df_lift_final.plot(x='decile', y='cum_lift')
plt.show()


# By using the model, we can reduce the cost of telemarketing by 50% by only targeting the specific prospects for selling the products. Without using the model, we had to target all the customer randomly, which results in lot time, effort and cost.
