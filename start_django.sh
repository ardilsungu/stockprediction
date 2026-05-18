#!/bin/bash
cd "$(dirname "$0")/backend"
source ../venv/bin/activate
PYTHONHASHSEED=0 python manage.py runserver
