
# coding: utf-8

# ## The Dataset
# The reason for the premature leaving of the best and most experienced employees. 
# 
# ### Aim 
# Try to predict which valuable employees will leave next. 
# 
# ##### Fields in the dataset include:
# <ol>
# <li>Satisfaction Level</li>
# <li>Last evaluation</li>
# <li>Number of projects</li>
# <li>Average monthly hours</li>
# <li>Time spent at the company</li>
# <li>Whether they have had a work accident</li>
# <li>Whether they have had a promotion in the last 5 years</li>
# <li>Departments (column sales)</li>
# <li>Salary</li>
# <li>Whether the employee has left</li>
# </ol>

# # Explorative Data Analysis

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-colorblind')
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[106]:


df = pd.read_csv('HR_comma_sep.csv')
df.head()


# In[107]:


df.keys()


# In[108]:


df.groupby('left').describe()


# In[109]:


sns.pairplot(df, hue='left')


# In[110]:


df.info()


# In[111]:


sns.jointplot(x=df['last_evaluation'], y=df['satisfaction_level'], kind='kde')


# In[112]:


sns.lmplot(data=df, x='satisfaction_level', y='average_montly_hours', size=12, hue='left')


# In[113]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)


# In[114]:


g = sns.FacetGrid(df, col = 'left', size=5)
g.map(sns.boxplot, 'time_spend_company')


# In[115]:


#time spend with promotion
plt.figure(figsize=(14,8))
sns.barplot(x='time_spend_company', y = 'left', hue = 'promotion_last_5years', data = df)


# # Data Transformation

# In[116]:


sal_dummy = pd.get_dummies(df['salary'])
df_new = pd.concat([df, sal_dummy], axis=1)


# In[117]:


df_new.drop('salary', axis=1, inplace=True)


# In[118]:


df_new.head()


# In[119]:


X = df_new.drop(['sales', 'left', 'high'], axis=1)
y = df_new['left']


# # Data Predictions
# 
# ## 1. Random Forest Classifier

# In[120]:


from sklearn.cross_validation import train_test_split


# In[121]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[122]:


from sklearn.ensemble import RandomForestClassifier


# In[123]:


rfc = RandomForestClassifier(n_estimators=100)


# In[124]:


rfc.fit(X_train, y_train)


# In[125]:


pred = rfc.predict(X_test)


# In[126]:


from sklearn.metrics import confusion_matrix, classification_report


# In[127]:


print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# In[128]:


rfc_score_train = rfc.score(X_train, y_train)
print('RFC Train Score:',rfc_score_train)
rfc_score_test = rfc.score(X_test, y_test)
print('RFC Test Score:',rfc_score_test)


# ## 2. K Nearest Neighbors Classifier

# In[129]:


from sklearn.neighbors import KNeighborsClassifier


# In[130]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[131]:


knn.fit(X_train, y_train)


# In[132]:


knn_pred = knn.predict(X_test)


# In[133]:


print(confusion_matrix(y_test, knn_pred))
print('\n')
print(classification_report(y_test, knn_pred))


# In[134]:


knn_score_train = knn.score(X_train, y_train)
print('KNN Train Score:', knn_score_train)
knn_score_test = knn.score(X_test, y_test)
print('KNN Test Score:', knn_score_test)


# ## 3. Logistic Regression

# In[135]:


from sklearn.linear_model import LogisticRegression


# In[136]:


lreg = LogisticRegression()


# In[137]:


lreg.fit(X_train, y_train)


# In[138]:


reg_pred = lreg.predict(X_test)


# In[139]:


print(confusion_matrix(y_test, reg_pred))
print('\n')
print(classification_report(y_test, reg_pred))


# In[140]:


lreg_score_train = lreg.score(X_train, y_train)
print("Logistic Regression Train Score:", lreg_score_train)
lreg_score_test = lreg.score(X_test, y_test)
print('Logistic Regression Test Score:', lreg_score_test)


# # Relationships Analyzed
# #### Exploring Further Relationships
# <ol>
# <li>Satisfaction level vs average monthly hours </li>
# <li>Department with highest turnover</li>
# <li>last evaluation vs satisfaction level</li>
# <li>Deeper analysis of regretted attrition:
#     <ul>
#     <li>Did they have a recent promotion </li>
#     <li>Did they spend too much time at work</li>
#     <li>Work accident</li>
#     <li>Number of projects</li>
#     <li>Last evaluations</li>
#     </ul>    
# </li> 
# </ol>    

# In[141]:


import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm


# In[142]:


df = pd.read_csv('HR_comma_sep.csv')
df.columns
df = df.rename(columns={'sales': 'department'})


# In[143]:


#isolate the data of those that have quit (left)
dfleft = df[df["left"]==1]
dfstay = df[df["left"]==0]


# In[144]:


#Did salary have an influence on the quitting rate of employees?
dfsalaryleft = dfleft.salary.value_counts().sort_index()
dfsalarystay = dfstay.salary.value_counts().sort_index()


# In[145]:


#A look into the distribution rates of Salary vs Satisfaction
fig = plt.figure(figsize=(10,20)) 

plt.subplot2grid((8,2),(0,0), colspan=2)
df.satisfaction_level[df.salary == 'low'].plot(kind='kde')    
df.satisfaction_level[df.salary == 'medium'].plot(kind='kde') 
df.satisfaction_level[df.salary == 'high'].plot(kind='kde') 
# plots an axis lable
plt.tight_layout()
plt.xlabel("Satisfaction_level")    
plt.title("Satisfaction level Distribution within salary")
# sets our legend for our graph.
plt.legend(('Low', 'Medium','High'),loc='best') 

plt.subplot2grid((6,2),(1,0), colspan=2)
dfleft.satisfaction_level[dfleft.salary == 'low'].plot(kind='kde')    
dfleft.satisfaction_level[dfleft.salary == 'medium'].plot(kind='kde') 
dfleft.satisfaction_level[dfleft.salary == 'high'].plot(kind='kde') 
# plots an axis lable
plt.xlabel("Satisfaction_level")    
plt.title("Satisfaction level Distribution within salary (Left)")
# sets our legend for our graph.
plt.legend(('Low', 'Medium','High'),loc='best')


# In[146]:


#An analysis of the Employed vs Left employees grouped by Department
depleft = df[['department','left']]
depleft1 = depleft.groupby(['department', 'left'])
ind = np.arange(20)                # the x locations for the groups
width = 0.35                      # the width of the bars
#print (depleft1['left'].value_counts())

#Without the hierarchical indexing
#print ("[Unstacking]")
depleft1sum = depleft1['left'].value_counts().unstack()
#print (depleft1sum)

ax = depleft1sum.plot(kind = 'bar', width = 1,  align='center')

ax.set_xlim(width)
ax.set_ylim(0,3300)
xTickMarks = ['','IT','','RandD','Accounting','','','HR','Management','','Marketing','','Product MMG','','','Sales','','Support','','Technical']
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
ax.set_title('Employed vs Quit Counts Grouped by Department')
plt.setp(xtickNames, rotation=45, fontsize=10)
plt.legend(('Stayed', 'Quit'),loc='best') 

#plt.ylabel(‘Num’)
plt.grid()
plt.show()


# In[147]:


#Analysis of regretted attrition 
#last evalutions > .85, number_project > 5, promotion_last_5years = 1
#df[(df["LATITUDE"] == '37.869058') | (df["LONGITUDE"] == '-122.270455')]

retention = dfleft[(dfleft["number_project"] > 5) | (dfleft["promotion_last_5years"] == 1)| (dfleft["last_evaluation"] > .85)]
retention = retention[['satisfaction_level','last_evaluation','promotion_last_5years','salary','number_project']]
retention.head()


# In[148]:


# Generate some test data
x = retention.satisfaction_level
y = retention.last_evaluation

heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.xlabel('Satisfaction_level')
plt.ylabel('last_evaluation')
plt.show()


# In[149]:


dummies=pd.get_dummies(df['department'],prefix='sales')
df=pd.concat([df,dummies],axis=1)
df.drop(['department'],axis=1,inplace=True)
df.head(10)


# In[150]:


dfdeptleft = dfleft.department.value_counts().sort_index().to_frame()
dfdeptstay = dfstay.department.value_counts().sort_index()
dfdeptleft = df.left.value_counts().sort_index().to_frame()
dfdeptleft.head()


# In[151]:


gbl_df = dfdeptleft
gbl_df.plot(kind = 'bar')
plt.show()

