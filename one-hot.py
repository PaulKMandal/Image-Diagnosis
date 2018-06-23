import csv

labels = []

with open('Data_Entry_2017.csv') as csvfile:
    readData = csv.reader(csvfile, delimiter = ',')

    
    for row in readData:
        label = row[1]
        
        labels.append(label)
        
labels = labels[1:]    
split_labels = [items.split('|')[0] for items in labels]
print(split_labels)

import numpy
import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

int_array = numpy.hstack(split_labels)
print(int_array)
values = array(split_labels)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

