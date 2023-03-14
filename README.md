# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

We create a simple dataset with one input and one output. This data is then divided into testing and training sets for our Neural Network Model to train and test on. The NN Model contains input layer, 2 nodes/neurons in the hidden layer which is then connected to the final output layer with one node/neuron. The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. The model is then trained for 2000 epochs.
We then perform an evaluation of the model with the test data. An user input is then predicted with the model. Finally, we plot the Training Loss VS Iteration graph for the given model.

## NEURAL NETWORK MODEL

![nn](https://user-images.githubusercontent.com/75234991/188797088-90a2a2ff-a38d-431f-9cce-f2f76358819b.svg)

## DESIGN STEPS

### Step 1:

Load the dataset.

### Step 2:

Split the dataset into training and testing data.

### Step 3:

Create MinMaxScalar object, fit the model and transform the data.

### Step 4:

Build the Neural Network Model and compile the model.

### Step 5:

Train the model with the training data.

### Step 6:

Plot the performance plot.

### Step 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('dataset').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])

df.head()

df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
df.dtypes

X=df[['X']].values
Y=df[['Y']].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=50)

scaler=MinMaxScaler()
scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

ai_brain=Sequential([
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=X_train_scaled,y=Y_train,epochs=20000)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test_scaled=scaler.transform(X_test)
ai_brain.evaluate(X_test_scaled,Y_test)

prediction_test=int(input("Enter the value to predict: "))
preds=ai_brain.predict(scaler.transform([[prediction_test]]))
print("The prediction for the given input "+str(prediction_test)+" is: "+str(preds))
```
## DATASET INFORMATION

<img width="81" alt="image" src="https://user-images.githubusercontent.com/75234991/187664392-e99a8824-e619-4818-80a7-ea250b3866b2.png">

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/75234991/187843085-4877d40f-fa95-4b7d-8930-5533e12886fa.png)

### Test Data Root Mean Squared Error

<img width="482" alt="image" src="https://user-images.githubusercontent.com/75234991/187843234-8bbb2a59-b725-4651-ac7b-667b0d1f361f.png">

### New Sample Data Prediction

<img width="348" alt="image" src="https://user-images.githubusercontent.com/75234991/187843282-09d51d46-d97c-4f8e-b7d4-8773466b4cbf.png">

## RESULT
Thus, a Simple Neural Network Regression Model is developed successfully.
