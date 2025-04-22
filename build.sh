#!/usr/bin/env bash

# Asegurar que el entorno está limpio
set -o errexit
set -o pipefail
set -o nounset

# Instalar portaudio para que PyAudio compile bien
apt-get update && apt-get install -y portaudio19-dev

# Instalar dependencias del proyecto
pip install --upgrade pip
pip install -r requirements.txt

# Si estás usando Django (opcional)
python manage.py migrate

