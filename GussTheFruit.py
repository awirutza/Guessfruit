import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#สร้าง Dataset
type_fruit = ['apple', 'orange'] 
weight = np.random.uniform(0,50,size=100) #กรัม
color = ['red','orange']
diameter = np.random.uniform(0,50, size=100) #ซม
taste = ['sweet','sour']

data = {
    'weight':weight,
    'color':random.choices(color,k=100),
    'diameter':diameter,
    'taste':random.choices(taste , k=100),
    'name_fruit':random.choices(type_fruit , k=100),
}

fruit_df  = pd.DataFrame(data)

# สร้างอินสแตนซ์ของ LabelEncoder
le = LabelEncoder()

fruit_df['color_encode'] = le.fit_transform(fruit_df['color'])
fruit_df['taste_encode'] = le.fit_transform(fruit_df['taste'])
fruit_df['name_fruit_encode'] = le.fit_transform(fruit_df['name_fruit'])

#กำหนด feature และ target
X = fruit_df[['weight','color_encode','diameter','taste_encode']] #feature
y = fruit_df[['name_fruit_encode']] #terget

#สร้าง Model
clf = DecisionTreeClassifier()

clf = clf.fit(X,y)
# Create a DataFrame for the prediction
sample_data = pd.DataFrame([{'weight': 15.258602, 'color_encode': 1, 'diameter': 4.7, 'taste_encode': 1}])

Prediction = clf.predict(sample_data)

print(Prediction)
