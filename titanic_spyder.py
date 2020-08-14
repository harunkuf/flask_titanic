import pandas as pd
import numpy as np

# machine learning
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('titanic_train.csv')
test = pd.read_csv('titanic_test.csv')
combine = pd.concat([train,test],ignore_index=True)


s = combine["Name"]
s = s.str.split(pat=",", expand = True)
s = s[1].str.split(pat=".", expand = True)
s= s[0]
combine["title"] = s


combine['title'] = combine['title'].str.replace('Mlle', 'Miss')
combine['title'] = combine['title'].str.replace('Ms', 'Miss')
combine['title'] = combine['title'].str.replace('Mme', 'Mrs')

rare_title = '|'.join(['Dona','Lady','the Countess','Capt','Col','Don', 
                'Dr','Major','Rev','Sir','Jonkheer'])

combine['title'] = combine['title'].str.replace(rare_title, 'raretitle')


s = combine["Name"]
s = s.str.split(pat=",", expand = True)[0]
combine["surname"] = s


combine["Fsize"] = combine["SibSp"] + combine["Parch"] + 1

combine["Family"] = combine.surname + "_" + combine.Fsize.map(str) 

def f(row):
    if row['Fsize'] == 1:
        val = "singleton"
    elif row["Fsize"] > 4:
        val = "large"
    elif row['Fsize'] < 5 and row['Fsize'] > 1:
        val = "small"
    return val


combine['FsizeD'] = combine.apply(f, axis=1)

combine["Deck"] = combine["Cabin"].str[0]

embarked_fare = combine[combine["PassengerId"] != 62]
embarked_fare = embarked_fare[embarked_fare["PassengerId"] != 830]

combine.loc[combine["PassengerId"] == 62,"Embarked"] = "C"
combine.loc[combine["PassengerId"] == 830,"Embarked"] = "C"

p1d = combine[((combine["Pclass"] == 3) & (combine["Embarked"] == "S"))]

combine["Fare"][1043] = p1d['Fare'].median()

ddf = combine.copy()

ddf.drop(['PassengerId','Name','Ticket','Cabin','Family','surname','Survived'], axis=1,inplace=True)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
X = imp.fit_transform(ddf)


X = pd.DataFrame(X).copy()
X.columns = ["Age","Embarked","Fare","Parch","Pclass","Sex","SibSp","title","Fsize","FsizeD","Deck"]

combine["Age"] = X["Age"]


combine["IsAdult"] = np.where(combine['Age'] < 18, '0', '1')


combine["IsMother"] = np.where((combine['Sex'] == "female") 
                             & (combine["Parch"] > 0) & (combine["Age"] > 18) 
                             & (combine["title"] != "Miss")
                             , '1', '0')

X = combine.copy()
X.drop(["Cabin","Embarked","Fare","Deck","Name","PassengerId","Ticket","surname"
           ,"Fsize","Family"], axis=1,inplace=True)

X = X[['Age', 'Sex', "Pclass", "Parch", "SibSp", "IsMother", "IsAdult", "title", "FsizeD", "Survived"]]
    

X['Sex'].replace(to_replace=['male','female'], value=[1,0],inplace=True)


from sklearn import preprocessing

Feature = X.copy()


labelEncoder = preprocessing.LabelEncoder() 

category_col =["title"]

mapping_dict ={} 
for col in category_col: 
    Feature[col] = labelEncoder.fit_transform(X[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 

def f2(col):
    if col['FsizeD'] == "singleton":
        val = 0
    elif col["FsizeD"] == "small":
        val = 1
    elif col['FsizeD'] == "large":
        val = 2
    return val

Feature['FsizeD'] = Feature.apply(f2, axis=1)
    


X_train = Feature[0:891].copy()
X_train.drop(["Survived"], axis = 1,inplace=True)
X_test = Feature[891:1309].copy()
X_test.drop(["Survived"], axis = 1,inplace=True)
Y_train = train['Survived'].values


LR = LogisticRegression(C = 0.1,solver='liblinear').fit(X_train,Y_train)


import pickle


pickle.dump(LR, open('model_lr.pkl','wb'))

model_lr = pickle.load(open('model_lr.pkl','rb'))




