import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Reading files
data = pd.read_csv('LOAN_DATA.csv')

data.describe()
print(data.head(10))

#checking null values in the dataset

print(data.isnull())

#since no null values we can proceed without data cleaning

#we know our target value is LOAN_STATUS so we now plot it

#seeing how many unique input categories are used
print(data.loan_status.unique())

#['PAIDOFF' 'COLLECTION' 'COLLECTION_PAIDOFF'] are the three unique values
#dropping loan id as it is useless
# dropping passed columns
data.drop(["Loan_ID"], axis=1, inplace=True)
print(data.head(10))


#plotting the data
fig = plt.figure(figsize=(5,5))
ax = sns.countplot(data.loan_status)
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
plt.show()


#These data points are the demographic information of the applicant.

fig, axs = plt.subplots(3, 2, figsize=(16, 15))
sns.distplot(data.age, ax=axs[0][0])
axs[0][0].set_title("Total age distribution across dataset")
sns.boxplot(x='loan_status', y='age', data=data, ax=axs[0][1])
axs[0][1].set_title("Age distribution by loan status")
sns.countplot(x='education', data=data, ax=axs[1][0])
axs[1][0].set_title("Education count")
for t in axs[1][0].patches:
    if (np.isnan(float(t.get_height()))):
        axs[1][0].annotate('', (t.get_x(), 0))
    else:
        axs[1][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

sns.countplot(x='education', data=data, hue='loan_status', ax=axs[1][1])
axs[1][1].set_title("Education by loan status")
for t in axs[1][1].patches:
    if (np.isnan(float(t.get_height()))):
        axs[1][1].annotate('', (t.get_x(), 0))
    else:
        axs[1][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

axs[1][1].legend(loc='upper right')
sns.countplot(x='Gender', data=data, ax=axs[2][0])
axs[2][0].set_title("# of Gender")
for t in axs[2][0].patches:
    if (np.isnan(float(t.get_height()))):
        axs[2][0].annotate('', (t.get_x(), 0))
    else:
        axs[2][0].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

sns.countplot(x='Gender', data=data, hue='education', ax=axs[2][1])
axs[2][1].set_title("Education of the gender")
for t in axs[2][1].patches:
    if (np.isnan(float(t.get_height()))):
        axs[2][1].annotate('', (t.get_x(), 0))
    else:
        axs[2][1].annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

plt.show();

#CHANGING CATEGOICAL DATA INTO NUMERICAL DATA
#change= {"PAIDOFF": 1, "COLLECTION": 2, "COLLECTION_PAIDOFF": 2 }

#data['loan_status_trgt'] = data['loan_status'].map(change)

data['loan_status'].replace('PAIDOFF',0, inplace=True)
data['loan_status'].replace('COLLECTION',1, inplace=True)
data['loan_status'].replace('COLLECTION_PAIDOFF',1, inplace=True)
print(data.head(5))

#loan status has been changed to categorical

#now we convert gender to categorical
#change= {"male": 0, "female": 1}

data['Gender'].replace('male',0, inplace=True)
data['Gender'].replace('female',1, inplace=True)
#data['Gender_trgt'] = data['Gender'].map(change)

print(data.head(5))
#we convert education and Gender to the dummy variables.
#dummies = pd.get_dummies(data['education']).rename(columns=lambda x: 'is_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['education'],  axis=1)

#dummies = pd.get_dummies(data['Gender']).rename(columns=lambda x: 'is_' + str(x))
#data = pd.concat([data, dummies], axis=1)
#data = data.drop(['Gender'], axis=1)

data['education'].replace('college',4, inplace=True)
data['education'].replace('High School or Below',3, inplace=True)
data['education'].replace('Bechalor',5, inplace=True)
data['education'].replace('Master or Above',6, inplace=True)
print(data.head(2))

data.drop(['effective_date','due_date','paid_off_time','past_due_days'], axis=1, inplace=True)
#dropping dummy variables to avoid dummy trap


#dummy_var = ['is_female', 'is_Master or Above']
#data = data.drop(dummy_var, axis = 1)

print(data.head(10))

#giving input and output to mpodel
x = data.drop(['loan_status'],axis = 1)
y = data['loan_status']
print(y.head(2))
#importing models
#Train-Test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size = 0.3, random_state = 0)

#logisitic regression
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(x_train , y_train)
lgr_score_train=lgr.score(x_train , y_train)
print("Training Score using Logistic Regression :",lgr_score_train )
lgr_score_test = lgr.score(x_test , y_test)
print("Testing Score Logistic Regression : ",lgr_score_test)
#print ("Train set :", y_train.head())
#print("Test set :" , x_test.head())

# now using decision tree
from sklearn import tree
dst = tree.DecisionTreeClassifier()
dst.fit(x_train, y_train)
dst_score_train = dst.score(x_train, y_train)
print("Training score using Descion Tree: ",dst_score_train)
dst_score_test = dst.score(x_test, y_test)
print("Testing score using Descion Tree: ",dst_score_test)