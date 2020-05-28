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
numberComments = data[:, 5]
likes          = data[:, 6]
shared         = data[:, 7]

#Adjustment of non-binary classification attributes
ct         = ColumnTransformer([('binario', OneHotEncoder(), [0, 1])], remainder='passthrough')
attributes = ct.fit_transform(attributes).toarray()

#Model comments
modelComments = Sequential()
modelComments.add(Dense(units=4, activation='relu'))
modelComments.add(Dense(units=1, activation='linear'))
modelComments.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

#Model likes
modelLikes = Sequential()
modelLikes.add(Dense(units=4, activation='relu'))
modelLikes.add(Dense(units=1, activation='linear'))
modelLikes.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

#Model Shared
modelShared = Sequential()
modelShared.add(Dense(units=4, activation='relu'))
modelShared.add(Dense(units=1, activation='linear'))
modelShared.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

modelComments.fit(attributes, numberComments, batch_size=30, epochs=1000)
modelLikes.fit(attributes, likes, batch_size=30, epochs=1000)
modelShared.fit(attributes, shared, batch_size=30, epochs=1000)

#Inputs
postType = int(input('Informe o número de tipo da postagem: Foto[0] | Link[1] | Status[2] | Video[3]: '))
month    = int(input('Mês: '))
day      = int(input('Dia da semana: D[1] | S[2] | T[3] | Q[4] | Q[5] | S[6] | S[7]: '))
hour     = int(input('Hora: '))
pay      = int(input('Pago: SIM[1] | NÃO[0]: '))

#Predicting comments
valuesComments = np.array([
    [postType, month, day, hour, pay]
])
valuesComments = ct.transform(valuesComments).toarray()

averageComments = modelComments.predict(valuesComments)

#Predicting Likes
valuesLikes = np.array([
    [postType, month, day, hour, pay]
])
valuesLikes = ct.transform(valuesLikes).toarray()

averageLikes = modelLikes.predict(valuesLikes)

#Predicting Shared
valuesShared = np.array([
    [postType, month, day, hour, pay]
])
valuesShared = ct.transform(valuesShared).toarray()

averageShared = modelShared.predict(valuesShared)

#Showing Averages
print('Média Comentários: ', int(averageComments[0]))
print('Média Likes: ', int(averageLikes[0]))
print('Média Compartilhamentos: ', int(averageShared[0]))

