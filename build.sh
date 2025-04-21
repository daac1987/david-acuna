#!/usr/bin/env bash

# Usar pyenv para fijar versión de Python
pyenv install 3.10.11
pyenv global 3.10.11

# Continuar con instalación normal
pip install --upgrade pip
pip install -r requirements.txt

# Comando para migraciones (si aplica)
python manage.py migrate

# O cualquier otro comando que usás
