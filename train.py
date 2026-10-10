import pandas as pd 
import joblib 
import json 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier


print('Loading data Successfully ')
df = pd.read_excel('Telecom_Tower_Failure_Dataset_10000-1.xlsx')
X = df[['Temperature_C','Battery_Voltage',        'Power_Consumption_W',
        'Signal_Strength_Percent',
        'Fan_Speed_RPM',
        'Humidity_Percent',
        'Traffic_Load',
        'Tower_Age_Years'
]]
y = df['Failure_Within_48Hrs']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 42, test_size = 0.25)

print('Training Model')
model = RandomForestClassifier(n_estimators = 100 , random_state=42)

model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)

metric = {accuracy:'accuracy'}
joblib.dump(model,'tower.pkl')

with open('metric.json','w') as f :
	json.dump(metric, f , indent=4 )

print('Training Completed Successfully ')