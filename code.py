import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score,confusion_matrix,classification_report
import pandas as pd
from google.colab import files
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
uploaded=files.upload()

df=pd.read_csv('diabetes_prediction_dataset.csv')
print(df)
print(df.describe())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.index)

#encoding gender
la=LabelEncoder()
df['gender']=la.fit_transform(df['gender'])
print(df['gender'])

#bar graph of smoking history counts
df['smoking_history'].value_counts().plot(kind='bar')

#dropping records where smoking history hs No Info
df=df[df['smoking_history']!='No Info']
print(df.index)

#encoding smoking history
df['smoking_history']=la.fit_transform(df['smoking_history'])
print(df['smoking_history'])
print(df['smoking_history'].max())

print(df['smoking_history'].unique())
#print count of 'ever' occurance in smoking history
print(df['smoking_history'].value_counts())

print(df.index)

#outliers
a=['HbA1c_level','blood_glucose_level','age','bmi']
for i in a:
  np.random.seed(42)
  data=df[i]
  plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
  plt.title(i)
  plt.show()


#removing outliers from
a=['HbA1c_level','blood_glucose_level','age','bmi']
def RemoveOutlier(column_name, dataframe):
      Q1 = dataframe[column_name].quantile(0.25)
      Q3 = dataframe[column_name].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      newdf = dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]
      return newdf
for i in a:
  df = RemoveOutlier(i, df)


#removed outliers
#boxplot of HbA1c_level
np.random.seed(42)
data=df['HbA1c_level']
plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
plt.title("HbA1c_level")
plt.show()
#boxplot of blood_glucose_level
np.random.seed(42)
data=df['blood_glucose_level']
plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
plt.title("blood_glucose_level")
plt.show()
#boxplot of age
np.random.seed(42)
data=df['age']
plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
plt.title("age")
plt.show()
#boxplot of bmi
np.random.seed(42)
data=df['bmi']
plt.boxplot(data,vert=True,labels=['Boxplot example'],patch_artist=True)
plt.title("bmi")
plt.show()

x=np.array(df.drop(['diabetes'],axis=1))
y=np.array(df['diabetes'])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)
print(y_train)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Intercept : ",model.intercept_)
print("Slope : ",model.coef_[0])
print("predicted y_train samples : ",y_pred[0:6])
print("actual y_train samples : ",y_test[0:6])


print(model.score(x_test,y_test))

#heatmap of d using imshow()
d=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(16,16))
plt.imshow(d,cmap='Greens')
plt.colorbar()
plt.show()

print(classification_report(y_test,y_pred))

#print s graph
plt.figure(figsize=(16,16))
plt.scatter(y_test,y_pred,color='red')
plt.plot(y_test,y_test,color='blue')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.show()

X_train_5 = x_train[:, [5]]
X_test_5  = x_test[:10, [5]] # Select only the first 10 records

model_5 = LogisticRegression()
model_5.fit(X_train_5, y_train)

y_prob_5 = model_5.predict_proba(X_test_5)[:, 1]
x5 = X_test_5.ravel()
sorted_idx = np.argsort(x5)

plt.figure(figsize=(8,6))
plt.scatter(x5, y_test[:10], alpha=0.5, label='Actual Data') # Slice y_test for first 10 records

plt.plot(x5[sorted_idx], y_prob_5[sorted_idx],
         label='Sigmoid Curve', linewidth=2)

plt.xlabel('Feature 5 (BMI)')
plt.ylabel('Probability of Diabetes')
plt.title('1D Logistic Regression (Feature 5 only) - First 10 Records')
plt.grid(True)
plt.legend()
plt.show()
