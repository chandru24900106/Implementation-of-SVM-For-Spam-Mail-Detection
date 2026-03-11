# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1. **Import necessary libraries** such as `pandas`, `CountVectorizer`, `SVC`, and metrics from `sklearn` for data handling, feature extraction, model building, and evaluation.

2. **Load and preprocess the dataset** using `pandas.read_csv()` with correct encoding, then extract input features (`x = data['v2']`) and target labels (`y = data['v1']`).

3. **Split the dataset** into training and testing sets using `train_test_split()` to prepare data for model training and validation.

4. **Convert text data into numeric form** using `CountVectorizer()` to transform the text messages into word count vectors (Bag-of-Words model).

5. **Train and test the SVM model** using `svc.fit(x_train, y_train)` and `svc.predict(x_test)`, then evaluate performance with accuracy, confusion matrix, and classification report.




## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: chandru v
RegisterNumber: 212224230043
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
*/
```

## Output:


### data


<img width="961" height="607" alt="Screenshot 2025-11-09 203843" src="https://github.com/user-attachments/assets/6835da30-8632-4b7d-af36-8f01780214c2" />

### data.shape()

<img width="307" height="49" alt="Screenshot 2025-11-09 203851" src="https://github.com/user-attachments/assets/1c78d3c1-ec89-4770-95d5-973bc6708335" />

### x.shape()

<img width="150" height="52" alt="Screenshot 2025-11-09 203859" src="https://github.com/user-attachments/assets/ca05d561-2d6e-41e3-b1fc-c1835e7ca92c" />


### y.shape()  

<img width="178" height="46" alt="Screenshot 2025-11-09 203903" src="https://github.com/user-attachments/assets/cb5aa351-ebb1-4f88-a867-2a8ed3a90972" />

### x_train

<img width="1061" height="158" alt="Screenshot 2025-11-09 203919" src="https://github.com/user-attachments/assets/5b192508-a231-4ac8-ba13-e2bf4e5e31b7" />

### x_train.shape()

<img width="152" height="55" alt="Screenshot 2025-11-09 203928" src="https://github.com/user-attachments/assets/6171e11b-ff3b-4b91-8b15-08e255402092" />


### y_pred

<img width="766" height="58" alt="Screenshot 2025-11-09 203935" src="https://github.com/user-attachments/assets/58af4837-5987-4860-bad8-585f58ae4819" />


### acc (accuracy)

<img width="268" height="47" alt="Screenshot 2025-11-09 203940" src="https://github.com/user-attachments/assets/b7180620-567e-421c-b494-41d724513a55" />


### con (confusion matrix)

<img width="189" height="78" alt="Screenshot 2025-11-09 203945" src="https://github.com/user-attachments/assets/a1192bef-3d09-4324-8c05-abec65eaf371" />


### cl (classification report)

<img width="774" height="240" alt="Screenshot 2025-11-09 203955" src="https://github.com/user-attachments/assets/2bb32ff3-7c92-44d3-9585-a187b5327329" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
