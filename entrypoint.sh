#!/bin/sh
echo "Running app in debug mode!"
export FLASK_APP=api.py
flask run --host=0.0.0.0 --port 8180
