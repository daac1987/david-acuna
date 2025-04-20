from django.shortcuts import render,redirect
from django.http import HttpResponse
import speech_recognition as sr
from .forms import chatForms,formulario
# Create your views here.

def inicio(request):
    return render(request, 'paginas/index.html')
    
def chat(request): 
    return render(request, 'paginas/chat.html')

def proyectos(request): 
    return render(request, 'paginas/proyectos.html')

def nosotros(request): 
    return render(request, 'paginas/nosotros.html')

def vozToTexto(request):

    r = sr.Recognizer()
    r.energy_threshold = 1000.
    r.dynamic_energy_threshold=False

    mi = sr.Microphone(device_index=0)
    with mi as micro:
      print("Micrófono")
      audio = r.listen(micro)

    try:
          texto = r.recognize_google(audio,language='ES')
    except sr.UnknownValueError:
        texto=("No se pudo reconocer el audio")
    except sr.RequestError as e:
          texto=("Error al solicitar resultados al servicio de reconocimiento de voz de Google; {}".format(e))

    context = {
       'texto': texto
    } 
    print(texto)
    return render(request,"paginas/chat.html",context)

def nuevo(request):
    if request.method == 'POST':
          form = chatForms(request.POST)
          if form.is_valid():
            # Procesar los datos del formulario y realizar acciones
            preguntaRealizada = form.cleaned_data['pregunta']
            respuestaBot = generarRespuesta(preguntaRealizada)
            #respuesta = generarRespuesta(texto)
            # Puedes guardar los datos en una base de datos o realizar otras acciones
            # Redirigir a otra página después de procesar el formulario
            if respuestaBot is not None:
                print(preguntaRealizada + " forms " + respuestaBot)
            else:
                print(preguntaRealizada + " forms " + "Respuesta no disponible")
            return render(request,"paginas/chat.html",{'preguntaRealizada': preguntaRealizada, 'respuestaBot':respuestaBot})
    else:
         form = chatForms()   
    return render(request,"paginas/chat.html")

def meForms(request):
    if request.method == 'POST':
          form = formulario(request.POST)
          if form.is_valid():
            # Procesar los datos del formulario y realizar acciones
            nombre = form.cleaned_data['nombre']
            correo = form.cleaned_data['correo']
            telefono = form.cleaned_data['telefono']
            comentario = form.cleaned_data['comentario']
            # Puedes guardar los datos en una base de datos o realizar otras acciones
            # Redirigir a otra página después de procesar el formulario
            print(nombre +','+ telefono+" ,"+correo+" ,"+comentario)
            return render(request,"paginas/nosotros.html")
    else:
         form = formulario()   
    return render(request,"paginas/nosotros.html")


def generarRespuesta(pregunta):
    import nltk #biblioteca que da herramientas para el procesamiento del lenguaje natural.
    #from nltk.stem.lancaster import LancasterStemmer
    #stemmer = LancasterStemmer() # en este proceso se reducen la palabras a su forma base o raiz.
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer('spanish')  # Crear el stemmer en español
    from collections import Counter
    import numpy #manejo de matrices y arreglos multidimensionales.
    import random
    import pickle #hace el proceso de serializacion y deserializacion, que es el proceso de combertir un objeto a una secuencia de bytes, que en este caso es guardado en un archivo. 
    import json

    import spacy
    nlp = spacy.load('es_core_news_sm')
    #from spacy.symbols import ORTH
    from tensorflow.python.keras.models import load_model
    import os

    # Obtiene la ruta del directorio actual del archivo actual ('nuevo.py')
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Retrocede cuatro niveles para llegar al directorio raíz del proyecto
    static_directory = os.path.join(current_directory, 'static')

    # Construye la ruta completa al archivo 'intents.json' dentro de 'static'
    file_path = os.path.join(static_directory, 'intents.json')


    #cargamos el json en un oobjeto 
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)
  
    #manejo de errores, si encuentra el archivo data.pickle se cargan los datos en las vairables words, labels, training, output mediante la funcion pickle.load(file)
    with open('data.pickle', 'rb') as file:
         words, labels, training, output = pickle.load(file)

    model = load_model('model.tflearn')#cargamos el modelo

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

    def chat(s):
            
            # Predicción del modelo
            input_bag,validadacion = bag_of_words(s, words)#llama la funcion bag_of_words que devuelve la bolsa de datos y el boolean para ver si se encontraron coincidencias con los patrones.
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
    
                return random.choice(responses)

    respuesta = chat(pregunta)  
    if respuesta:
        return respuesta
    else:
        # Si no hay respuesta válida, retornamos un valor predeterminado
        return "No puedo entender tu pregunta. Por favor, reformúlala." 
   
                      

