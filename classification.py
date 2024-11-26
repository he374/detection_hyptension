import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from reader import read_csv



data = read_csv()


i1 = []
i2 = []
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []
c6 = []
c7 = []
c8 = []
c9 = []
c10 = []
l1 = []
l2 = []


for i in range(0,85):
    xs = data[i] 
    l1.append(xs[-1])
    l2.append(xs[-2])
    m = xs[0]
    i1.append(m[0][0])
    i2.append(m[0][1])
    c1.append(m[1][0][0])
    c2.append(m[1][0][1])
    c3.append(m[1][0][2])
    c4.append(m[1][0][3])
    c5.append(m[1][0][4])
    c6.append(m[1][1][0])
    c7.append(m[1][1][1])
    c8.append(m[1][1][2])
    c9.append(m[1][1][3])
    c10.append(m[1][1][4])




df = pd.DataFrame({'I1': i1, 'I2' : i2, 'C1' :c1, 'C2' :c2, 'C3' :c3, 'C4' :c4, 'C5' :c5, 'C6' :c6, 'C7' :c7, 'C8' :c8, 'C9' :c9, 'C10' :c10, 'labeld' :l1, 'labelm':l2}, columns = ['I1', 'I2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'labeld', 'labelm'])


x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state= 42)


print(x_train.shape)
print(x_test.shape)

model = LogisticRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)
print(predictions)
print(y_test)

print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))
