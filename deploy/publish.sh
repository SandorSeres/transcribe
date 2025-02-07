#!/bin/bash

# Ellenőrizzük, hogy az ngrok telepítve van
if ! command -v ngrok &> /dev/null
then
    echo "Az ngrok nem található. Kérlek telepítsd először az ngrok alkalmazást!"
    exit 1
fi

# A FastAPI szerver indítása (feltételezve, hogy main.py-ben van az 'app')
echo "FastAPI szerver indítása a 8000-as porton..."
#uvicorn main:app --host 127.0.0.1 --port 8000 &


FASTAPI_PID=$!

# Rövid várakozás, hogy a szerver elinduljon
sleep 3

# ngrok tunnel indítása a 8000-as portra
echo "ngrok tunnel indítása..."
ngrok http 8000

# Amikor az ngrok befejeződik, leállítjuk a FastAPI szervert
echo "FastAPI szerver leállítása..."
kill $FASTAPI_PID

