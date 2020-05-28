import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

#Data search
csv = pd.read_csv('dados.csv', sep=';')

#Data treatment
le = LabelEncoder()
csv['Tipo'] = le.fit_transform(csv['Tipo'])
data = csv.values

#Separating attributes and classifiers
attributes     = data[:,:5]
classifiers    = data[:,5:]


#Adjustment of non-binary classification attributes
ct         = ColumnTransformer([('binario', OneHotEncoder(), [0, 1])], remainder='passthrough')
attributes = ct.fit_transform(attributes).toarray()

#Create Model
model = Sequential()
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='linear'))
model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

#Training Model Comments
model.fit(attributes, classifiers, batch_size=30, epochs=1000)

#Inputs
postType = int(input('Informe o número de tipo da postagem: Foto[0] | Link[1] | Status[2] | Video[3]: '))
month    = int(input('Mês: '))
day      = int(input('Dia da semana: D[1] | S[2] | T[3] | Q[4] | Q[5] | S[6] | S[7]: '))
hour     = int(input('Hora: '))
pay      = int(input('Pago: SIM[1] | NÃO[0]: '))

#Predicting comments
values = np.array([
    [postType, month, day, hour, pay]
])
values   = ct.transform(values).toarray()
averages = model.predict(values)

#Showing Averages
print('Média Comentários: ', int(averages[0][0]))
print('Média Likes: ', int(averages[0][1]))
print('Média Compartilhamentos: ', int(averages[0][2]))

