import nltk #biblioteca que da herramientas para el procesamiento del lenguaje natural.
#from nltk.stem.lancaster import LancasterStemmer
#stemmer = LancasterStemmer() # en este proceso se reducen la palabras a su forma base o raiz.
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')  # Crear el stemmer en español
from collections import Counter
import numpy #manejo de matrices y arreglos multidimensionales.
import tensorflow as tf # desarrolla y entrena modelos de aprendizaje automatico.
import random
import pickle #hace el proceso de serializacion y deserializacion, que es el proceso de combertir un objeto a una secuencia de bytes, que en este caso es guardado en un archivo. 
import json

import spacy
nlp = spacy.load('es_core_news_sm')
#from spacy.symbols import ORTH

from tensorflow.python.keras.models import Sequential #permite crear redes neuronales apilando varias capas.
from tensorflow.python.keras.layers import Dense #permite conectar una capa a la capa anterior, realiza la multiplicacion de matrices y aplica una funcion de suma ponderada de matrices.
from tensorflow.python.keras.optimizers import gradient_descent_v2 #se utilizan para optimizar los pesos y los sesgos en el proceso de entrenamiento.


import os

# Obtiene la ruta del directorio actual del archivo actual ('nuevo.py')
current_directory = os.path.dirname(os.path.abspath(__file__))

# Retrocede cuatro niveles para llegar al directorio raíz del proyecto
project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_directory))))

# Construye la ruta completa al archivo 'intents.json' dentro de 'static'
file_path = os.path.join(project_directory, 'sistema', 'codigo', 'static', 'intents.json')


#cargamos el json en un oobjeto 
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  
#manejo de errores, si encuentra el archivo data.pickle se cargan los datos en las vairables words, labels, training, output mediante la funcion pickle.load(file)
try:
    with open('data.pickle', 'rb') as file:
        words, labels, training, output = pickle.load(file)

#en este segmento si no se encuentra el archivo data.pickle se asignan los datos        
except FileNotFoundError:
    #declaracion de listas de datos
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # tokenize lo que hace es dividir un texto en palabras individuales para el analisis
            wrds = nltk.word_tokenize(pattern)
            #wrds = [token.text for token in pattern if pattern.isalpha]
            # añade a la lista de palabras
            words.extend(wrds)
            # la lista se agrega a docs_x
            docs_x.append(wrds)
            # agrega la etiqueta tag del json a la lista docs_y
            docs_y.append(intent['tag'])

        #verifica duplicados en la lista tag e igualmente agrega al final de la lista de no existir, haciendo que la clave tag sea unica. 
        if intent['tag'] not in labels:
          labels.append(intent['tag']) #agrega

    
    #primero se verifica que los simbolos de la lista no se incluyan despues las palabras que no son estos simbolos de la lista se les aplica la funcion de combertir a minusculas, luego el estem las combierta a su forma raiz. La palabras que pasen el proceso se incluyen en la lista final words.
    #words = [stemmer.stem(str(w).lower())for w in words if w not in [ '?','!','[',']','{','}','¡','$','€']]
    words = [stemmer.stem(str(w).lower()) for w in words if isinstance(w, str) and w not in [ '[', ']', '{', '}', '$', '€']]
    
    """for token in words :
      if token.is_alpha and str(token) not in ['[', ']', '{', '}', '$', '€']:
        lemma = token.lemma_.lower()
        stemmed_word = lemma if lemma != "-PRON-" else token.lower_
        words.append(stemmed_word)"""

    #set eliminara los duplicados, en list se guardara el resultado sin duplicados, sorted ordenara la lista en orden ascendente y el resultado se guardara en words. 
    words = sorted(list(set(words)))

    #ordena la lista labels en orden ascendete
    labels = sorted(labels)

    #declaracion de listas de datos
    training = []
    output = []

    #creacion de la lista output empty, donde cada elemento es un cero y su longitud es conforme a la lista labels. Su objetico es iterar una cantidad de veces si necesitar su valor. 
    output_empty = [0 for _ in range(len(labels))]

    #en el for interamos o recorremos sobre el elemento doc_x para obtener el indice representado con x y el valor con doc. 
    for x, doc in enumerate(docs_x):
        #creacion de listas
        bag = []
        #wrds = [stemmer.stem(str(w).lower()) for w in doc] #en la iteracion de doc combierte las palabras en minuscula y las lleva a la raiz, creando una nueva lista. 
        wrds = [stemmer.stem(str(w).casefold()) for w in doc]

        
        #words lista creada en la linea 36 que contiene palabras individuales y unicas de la lista para el analisis,
        for w in words: # esta parte del codigo lo que hara es verificar la presencia o ausencia de una palabra en el documento.
            #si la palabra en la iteracion de doc es igual a la palabra en la iteracion de words, entonces se agrega 1 a la posicion de esa palabra en la lista bag. 
            if w in wrds:
                bag.append(1)#presente
            else:
                bag.append(0)#no presente

        output_row = output_empty[:]#se crea un nueva lista copia de output_empty y este segmento [:] hace que tengo los mismos elementos pero con un diferente espacio de memoria.
        output_row[labels.index(docs_y[x])] = 1 #asigna el valor 1 al indice de output_row, se usa para construir el vector de salidad deseado.

        training.append(bag)#se agregan los vectores de caracteristicas a la lista
        output.append(output_row)#se agregan los vectores de etiquetas a la lista, igual a len(labels).

    training = numpy.array(training)#numpy.array()se devuelven matrices NumPy que permiten trabajar de forma eficiente los datos en forma de matrices multidimencionales.
    output = numpy.array(output)#numpy.array()se devuelven matrices NumPy que permiten trabajar de forma eficiente los datos en forma de matrices multidimencionales.

    # se guardo los datos de la lista en el documento data.pickle, el documento es de escritura binaria 
    with open('data.pickle', 'wb') as file:
        pickle.dump((words, labels, training, output), file)#serializa los datos y los guarda en el archivo 
        
    tf.compat.v1.reset_default_graph()# se utiliza para restuarar tensorflow a su estado inicial.

model = Sequential() #crea una instancia de un modelo secuencial vacio. Es una pila de capas que se conecta directamente con la anterior. 
model.add(Dense(10, input_shape=(len(training[0]),), activation='relu'))#crea una capa de 8 neuronas y como entrada el vector de longitud len(training[0]). relu se utiliza para no introducir no linealidades en la red neuronal.
model.add(Dense(len(output[0]), activation='softmax'))#agrega otra capa conectada a la anterior, como categorias diferentes en los datos de salida. softmax se utiliza como una distribucion de probabilidades sobre clases multiples, lo que permite que el modelo haga predicciones sobre clases multiples.


sgd = gradient_descent_v2.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)#crea una instancia con parametros, learning_rate  que establece la tasa de aprendizaje, decay con el decaimiento de la tasa de aprendizaje, momentum controla la inercia del optimizador y supera los minimos locales, nesterov para mejorar el desempeño de sgd.
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])# compila el modelo, loss='categorical_crossentropy' funcion de perdida adecuada para problemas de clasificacion, optimizer=sgd optimizador para ajustar los parametros del modelo de entrenamiento, metrics=['accuracy']especifica que se calcula la metrica de presicion durante el entrenamiento.

#entran los datos, training y output contienen los vectores NumPy con los datos de entrenamiento. epochs=1000 numero de veces que el modelo se entrenara. batch_size=8 tamaño del lote utilizado en cada iteracion del entrenamiento. verbose=1 muestra la informacion detallada del entrenamiento.
model.fit(training, output, epochs=1500, batch_size=8, verbose=1)
model.save('model.tflearn')# guarda el modelo en un archivo model.tflearn. Almacena pesos, configuraciones del modelo y permiter cargarlo.

#funcion para identificar la presencia o ausencia de palabras. s parametro recibido de palabra ingresada.

def bag_of_words(s, words):
    bag = [0] * len(words)  # inicializa la lista de palabras

    validacion = False # variable creada para valiadar si pregunta ingresada coincide con algun patron del archivo json 
    s_words = nltk.word_tokenize(s)  # divide la lista de palabras ingresadas 
    s_words = [token.text for token in nlp(s)]
    #s_words = [token.lemma_.lower() for token in nlp(s) if token.is_alpha]

    s_words = [stemmer.stem(word.lower()) for word in s_words]  # convierte a minúsculas y divide las palabras de s
   
    word_freq = Counter(s_words)  # cuenta la frecuencia de cada palabra en s_words

    if not any(word in words for word in s_words):  # verifica si ninguna palabra en s_words coincide con las palabras en "words"
       validacion = True # al no tener palabras en el patron devuelve verdadero
    else: # verifica
        for i, word in enumerate(words):  # iterar sobre las palabras en la lista "words"
           if word in word_freq:
             bag[i] = word_freq[word]  # asigna la frecuencia de la palabra correspondiente en "bag"

    return bag , validacion

#funcio para tener conversacion basica.

def chat():
    print('Hola, comenzamos nuestra conversacion.\n Recuerda que para abandonar el chat digita la palabra "Salir"')
    while True:
        inp = input('Tu: ')  # entrada de usuario
        if inp.lower() == 'salir':
            print('Nos vemos.')
            break

        # Predicción del modelo
        input_bag,validadacion = bag_of_words(inp, words)#llama la funcion bag_of_words que devuelve la bolsa de datos y el boolean para ver si se encontraron coincidencias con los patrones.
        if validadacion == True:  # Verifica si es una pregunta sin patrón
            print('No puedo entender tu pregunta. Por favor, reformúlala.')

        else:
            #model.predict se utiliza para realizar predicciones, con entradad de datos.  
            result = model.predict([input_bag])
            result_index = numpy.argmax(result)#aca se encuentra el indice con etiqueta con el mayor uso.
            tag = labels[result_index]#optiene la etiqueta correspondiente a la prediccion realizada. labels tiene la lista de los tag del documento json.

            for tg in data['intents']:#data definidad en la linea 17 contiene la informacio del json.
                if tg['tag'] == tag:# se busca coincidencia entre las etiquetas.
                    responses = tg['responses']#si tienen coincidencia se asigna el valor de respuesta a la variable responses.

            print("Bot: ", random.choice(responses))


chat()# llamado de la funcion.

