import json  
import numpy as np #2.1.3
from sklearn.model_selection import train_test_split #1.6.1

from keras import models #3.10.0
from keras import layers
from keras import regularizers
from sklearn.metrics import mean_absolute_error
import os

import matplotlib.pyplot as plt #3.10.3
'''
simplejson==3.20.1
scikit-learn==1.6.1
numpy==2.1.3
keras==3.10.0
matplotlib==3.10.3
tf==1.0.3
tf-nightly==2.20.0.dev20250619
'''
register_words = []
ds = []
X, Y = [], []

#--------------------------------------Область объявлений функций--------------------------------------------------
#Функция преобразования списка строк (sequences) в матрицу формой [длина sequences]x[количество слов в register_words]
def vectorize_sequences(sequences, register_words):
	result = np.zeros((len(sequences), len(register_words)))
	for i, string in enumerate(sequences):
		string = string.lower().replace(',',' ').replace('-',' ').replace('.',' ').replace('!',' ').replace('?',' ').replace('(',' ').replace(')',' ').replace('"',' ').split()
		for e in string:
			result[i, register_words[e]] = 1
	return result #из ['Я и ты','Ты и я'] в [[0,1,0,1,1,0...0],[0,1,0,1,1,0...0]]

#Лямбда функция очистки консоли
clear = lambda: os.system('cls')
#-------------------------------------------------------------------------------------------------------------------

#--------------------------------------Область подготовки данных--------------------------------------------------
with open('news_descriptions_GPT4o.json', 'r') as f:
	f = f.read()
	f = json.loads(f)
	f = [e[1] for e in list(f.items())] #из словаря в список
	for obj in f:
		register_words += obj['thinking'].lower().replace(',',' ').replace('-',' ').replace('.',' ').replace('!',' ').replace('?',' ').replace('(',' ').replace(')',' ').replace('"',' ').split()
		X.append(obj['thinking'])
		Y.append(obj['sentiment_score'])

register_words = set(register_words)
register_words = {index:value for index, value in enumerate(register_words)}
register_words_invers = {value:key for key, value in register_words.items()}
X = vectorize_sequences(X ,register_words_invers)
Y = [1 if i>0 else 0 for i in Y] #GPT-4 дал оценку данным от -1 до 1, преобразовываем оценки в строгий вид - 0, 1
Y = np.asarray(Y).astype('float32')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
#-------------------------------------------------------------------------------------------------------------------

#----------------------------------Область моделирования и обучения нейросети---------------------------------------
EPOCHS = 4
BATCH_SIZE = 512

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(len(register_words),)))
model.add(layers.Dense(16, activation='relu',))
model.add(layers.Dense(4, activation='relu',))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
clear()
#-------------------------------------------------------------------------------------------------------------------

#-----------------------------Область выгрузки файлов для дублирования модели нейросети-----------------------------
#Выгрузка Регистра слов (Это не DataSet или правильные метки предсказаний)
with open('register_words.json','w') as f:
	f.write(json.dumps(register_words))
	f.close()
#Сохранение полной модели keras (вместе с весами) уже обученной
model.save('model.keras')
#-------------------------------------------------------------------------------------------------------------------

#--------------------------------Область демонстрации результатов обучении нейросети--------------------------------
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Точность на тестовом образцу:', test_acc)
print('Потери на тестовом образце:', test_loss)

history_dict = history.history
epochs = range(1, len(history_dict['loss'])+1)

plt.plot(epochs, history_dict['accuracy'], 'bo', label='Точность на этапе обучения')
plt.plot(epochs, history_dict['val_accuracy'], 'b', label='Точность на этапе проверки')
plt.title('Точность на этапе проверки и обучения')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.plot(epochs, history_dict['loss'], 'bo', label='Потери на этапе обучения')
plt.plot(epochs, history_dict['val_loss'], 'b', label='Потери на этапе проверки')
plt.title('Потери на этапах проверки и обучения')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()
#-------------------------------------------------------------------------------------------------------------------





	
'''
news_collection = pd.read_parquet('RussianFinancialNews/news_collection.parquet', engine='pyarrow')
news_description = pd.read_parquet('RussianFinancialNews/news_descriptions/news_collection_old.parquet', engine='pyarrow')
print(news_description)
'''