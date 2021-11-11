import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

myfile = pd.read_csv("heart.csv")
x=myfile.drop(['target'],axis=1)
y=myfile['target']
x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(x,y, test_size = 0.20)
model = RandomForestClassifier(n_estimators = 25)
# model.fit(x_train, y_train)
# with open ('heart.pickle','wb') as f:
#     pickle.dump(model,f)
pickle1=open('heart.pickle','rb')
model=pickle.load(pickle1)
acc=model.score(x_test,y_test)
print('accuracy is',acc)
columns=['Age(greater than 15)', 'Sex(1 for male or 0 for female)', 'CP(0 or 1 or 2 or 3)', 'trestbps', 'chol', 'fbs(0 or 1)',
         'restecg(0 or 1)', 'thalach','exang(0 or 1)', 'oldpeak','slope(0 or 1 or 2)', 'ca(0 or 1 or 2 or 3 or 4) ','thal(0 or 1 or 2 or 3)']
user =[]
for i in columns:
  value=float(input(f"Enter the value for {i.upper()}"))
  user.append(value)
pred=model.predict(np.array(user).reshape(1,-1))[0]
print(pred)
if pred>=1:
  print("Heart Diseases")
else:
  print("NO Heart Diseases ")