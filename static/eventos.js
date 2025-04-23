
function reproducirTexto() {
   // Obtener el texto ingresado por el usuario
   var texto = document.getElementById('respuestaBotAudio').value;

   // Crear una nueva instancia de la API SpeechSynthesis
   var synthesis = window.speechSynthesis;

   // Verificar si el navegador admite la síntesis de voz
   if ('speechSynthesis' in window) {
       // Crear un objeto de síntesis de voz
       var utterance = new SpeechSynthesisUtterance(texto);

       // Establecer el idioma (opcional)
       utterance.lang = 'es'; // Código de idioma para español

       // Reproducir el texto en voz
       synthesis.speak(utterance);
   } else {
       alert('Tu navegador no admite la síntesis de voz.');
   }
}

function reproducirTextoPregunta() {
   // Obtener el texto ingresado por el usuario
   var texto = document.getElementById('usuarioPregunta').value;

   // Crear una nueva instancia de la API SpeechSynthesis
   var synthesis = window.speechSynthesis;

   // Verificar si el navegador admite la síntesis de voz
   if ('speechSynthesis' in window) {
       // Crear un objeto de síntesis de voz
       var utterance = new SpeechSynthesisUtterance(texto);

       // Establecer el idioma (opcional)
       utterance.lang = 'es'; // Código de idioma para español

       // Reproducir el texto en voz
       synthesis.speak(utterance);
   } else {
       alert('Tu navegador no admite la síntesis de voz.');
   }
}

