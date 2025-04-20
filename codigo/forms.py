from django import forms

class chatForms(forms.Form):
    pregunta = forms.CharField(label='pregunta', max_length=100)

class formulario(forms.Form):
    nombre = forms.CharField(label='nombre', max_length=40)
    telfono = forms.CharField(label='telefono', max_length=20)    
    correo = forms.EmailField(label='correo') 
    comentario = forms.CharField(label='comentario', widget=forms.Textarea)