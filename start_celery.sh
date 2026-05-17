#!/bin/bash
cd "$(dirname "$0")/backend"
source ../venv/bin/activate
PYTHONHASHSEED=0 celery -A config worker --loglevel=info
