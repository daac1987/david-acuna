import speech_recognition as sr

r = sr.Recognizer()
r.energy_threshold = 1000.
r.dynamic_energy_threshold=False

mi = sr.Microphone(device_index=0)

def procesoVoz():

  with mi as micro:
    print("Micrófono")
    audio = r.listen(micro)

    try:
        texto = r.recognize_google(audio,language='ES')

    except sr.UnknownValueError:
        texto=("No se pudo reconocer el audio")

    except sr.RequestError as e:
        texto=("Error al solicitar resultados al servicio de reconocimiento de voz de Google; {}".format(e))
    
  return texto

t = procesoVoz()

print(t)