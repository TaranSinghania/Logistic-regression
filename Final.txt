#importing all the packages
import numpy as np
from random import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

#defining the class and the functions
class LogisticRegression(object):
    
    def __init__(self,lr=0.0001,itern=40000,random_state=10):
        self.lr=lr
        self.itern=itern
        self.random_state=random_state
        
    def fit(self,X,y):
        
        gen=np.random.RandomState(self.random_state)
        self._w=gen.normal(loc=0.,scale=1.,size=1+X.shape[1])
        
        self.cost_=[]
        for _ in range(self.itern):
            output=self._sigmoid_activation(self.input(X))
            Error=y-output
            self._w[1:]+=self.lr*X.T.dot(Error)
            self._w[0]+=self.lr*Error.sum()
            cost=-y.dot(np.log(output))-(1-y).dot(np.log(1-output))
            self.cost_.append(cost)
        return self
        
    def  input(self,X):
        return np.dot(X,self._w[1:])+self._w[0]
    
    def _sigmoid_activation(self,z):
        return 1/(1+np.exp(-z))
    
    def predict(self,X):
        return np.where(self._sigmoid_activation(self.input(X))>=0.5,1,-1)

    def L1_loss(y_pred, y):
        return np.sum(np.abs(y_pred - y))


#Loading data
X=load_iris().data
df=pd.DataFrame(X,columns=load_iris().feature_names)
df['target']=load_iris().target
df['label']=df.apply(lambda x:load_iris()['target_names'][int(x.target)],axis=1)

#Normalizing data
X = (X -np.mean(X))/X.std()
y=np.where(df['label']=='setosa',0,1)

#Scaling data
scaler=StandardScaler()
X_std=scaler.fit_transform(X) 

#Splitting the data
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.2,random_state=10)

#Implementing on training set
LRC=LogisticRegression(lr=0.01,itern=100,random_state=10)
LRC.fit(X_train,y_train)

#Checking accuracy
y_pred=LRC.predict(X_test)
print('accuracy_score=%.3f'%accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print('')
print(cm)


#Plotting graph
plt.style.use('ggplot')
plt.plot(np.arange(1,len(LRC.cost_)+1),LRC.cost_,color='blue',marker='.')
plt.xlabel('epochs')
plt.ylabel('cost function')
plt.show()


