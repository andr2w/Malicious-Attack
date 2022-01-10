import torch
import pandas as pd

#data = pd.read_csv('data.csv')
#print(data.head())
#

text_data = pd.read_csv('data.csv')
text = text_data['Text']
label = text_data['Label']

print(text.head())
print(label.head())
