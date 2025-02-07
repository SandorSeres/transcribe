#!/bin/bash

# Ellenőrizzük, hogy van-e bemeneti kép
if [ "$#" -ne 1 ]; then
    echo "Használat: $0 <képfájl>"
    exit 1
fi

IMAGE_PATH="$1"

# Ellenőrizzük, hogy a fájl létezik-e
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Hiba: A fájl nem található: $IMAGE_PATH"
    exit 1
fi

# Kép Base64 kódolása és fájlba mentése
BASE64_FILE="image_base64.txt"
base64 -w 0 "$IMAGE_PATH" > "$BASE64_FILE"

# JSON fájl létrehozása
JSON_FILE="request.json"
cat <<EOF > "$JSON_FILE"
{
  "model": "ozbillwang/deepseek-janus-pro",
  "prompt": "Mi van a képen?",
  "stream": false,
  "images": ["$(cat $BASE64_FILE)"]
}
EOF

# API kérés küldése a Janus modellnek
RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d @"$JSON_FILE")

# Az eredmény szép formázása
echo "💡 Modell válasza:"
echo "$RESPONSE" | jq '.response'

# Ideiglenes fájlok törlése
rm -f "$BASE64_FILE" "$JSON_FILE"

