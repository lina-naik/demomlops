import pandas as pd 
import joblib 

print('model loaded')
model =  joblib.load('tower.pkl')

new_data = pd.DataFrame({'Temperature_C':[33.4],
        'Battery_Voltage':[53.33],
        'Power_Consumption_W':[736],
        'Signal_Strength_Percent':[69],
        'Fan_Speed_RPM':[2823],
        'Humidity_Percent':[25],
        'Traffic_Load':[1166],
        'Tower_Age_Years':[5]}
)

prediction = model.predict(new_data)

if prediction[0] ==1 :
	print('Tower is not Healthy')
else :
	print('Tower is  Healthy')