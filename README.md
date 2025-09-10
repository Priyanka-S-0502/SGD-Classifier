# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries and Load Data

2.Split Dataset into Training and Testing Sets

3.Train the Model Using Stochastic Gradient Descent (SGD)

4.Make Predictions and Evaluate Accuracy

5.enerate Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PRIYANKA S
RegisterNumber:212224040255  
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Name:PRIYANKA S")
print("Reg No:212224040255")
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Name:PRIYANKA S")
print("Reg No:212224040255")
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Name:PRIYANKA S")
print("Reg No:212224040255")
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```

## Output:

<img width="839" height="386" alt="Screenshot 2025-09-10 182213" src="https://github.com/user-attachments/assets/6a023a59-13d1-4786-b1e9-48c65fe56379" />


<img width="329" height="112" alt="Screenshot 2025-09-10 182223" src="https://github.com/user-attachments/assets/4037dafb-f366-40cf-8c67-8f2e884192a4" />




<img width="416" height="95" alt="Screenshot 2025-09-10 182232" src="https://github.com/user-attachments/assets/5e579313-dc49-47e7-9a3e-f71cd0e14c2b" />




<img width="390" height="117" alt="Screenshot 2025-09-10 182241" src="https://github.com/user-attachments/assets/30963c10-3702-4b68-84e0-501914697956" />


<img width="825" height="311" alt="Screenshot 2025-09-10 182251" src="https://github.com/user-attachments/assets/14b58b30-27e2-431f-a96c-5bcfd7eab888" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
