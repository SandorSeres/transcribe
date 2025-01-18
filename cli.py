import os
import asyncio
import httpx
import json
from pathlib import Path

API_URL = "http://localhost:8000/upload_ocr/"
API_GENERATE = "http://localhost:8000/generate/"
IMAGE_DIR = "images"
OUTPUT_DIR = "processed_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def process_image(image_path):
    """Feldolgoz egy képet a FastAPI végpont meghívásával."""
    try:
        print(f"🔄 Processing: {image_path}")

        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return

        with open(image_path, "rb") as f:
            image_data = f.read()

        timeout = httpx.Timeout(600)  # 60 másodperces timeout
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            files = {"file": (Path(image_path).name, image_data, "image/jpeg")}

            # OpenAI GPT-4o feldolgozás
            data_openai = {"taskType": "describe", "modelType": "openai", "modelName": "gpt-4o"}
            response_openai = await client.post(API_URL, files=files, data=data_openai)
            response_openai.raise_for_status()
            print(f"✅ OpenAI response received for {image_path}")

            result_openai = response_openai.json().get("extracted_text", "")

            # Ollama llava3.2-vision feldolgozás
            data_ollama = {"taskType": "describe", "modelType": "ollama", "modelName": "llama3.2-vision"}
            response_ollama = await client.post(API_URL, files=files, data=data_ollama)
            response_ollama.raise_for_status()
            print(f"✅ Ollama response received for {image_path}")

            result_ollama = response_ollama.json().get("extracted_text", "")

            # Kombinálás OpenAI GPT-4o segítségével
            prompt = (
                "Merge the following two image descriptions into a single, coherent, well-structured caption:\n\n"
                f"**Description 1 (GPT-4o):** {result_openai}\n\n"
                f"**Description 2 (Ollama Llava3.2):** {result_ollama}"
            )
            data_generate = {"query": prompt, "modelType": "openai", "modelName": "gpt-4o"}
            response_combined = await client.post(API_GENERATE, data=data_generate)
            response_combined.raise_for_status()
            print(f"✅ Combined text response received for {image_path}")

            final_text = response_combined.json().get("text", "")

            # Eredmény mentése fájlba
            output_path = os.path.join(OUTPUT_DIR, f"{Path(image_path).stem}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            print(f"✅ Processed successfully: {image_path} -> {output_path}")
    except httpx.TimeoutException:
        print(f"❌ Timeout error processing {image_path}")
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error {e.response.status_code} processing {image_path}: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error processing {image_path}: {e}")

import asyncio

CONCURRENT_TASKS = 1  # Maximum párhuzamos feldolgozások száma

async def process_with_semaphore(semaphore, image_path):
    """Egy kép feldolgozása a megadott szemináriummal."""
    async with semaphore:  # Limitáljuk az egyidejű feldolgozásokat
        await process_image(image_path)

async def main():
    """Az összes kép feldolgozása az IMAGE_DIR könyvtárban, korlátozott párhuzamossággal."""
    images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        print("❌ No images found in the directory.")
        return

    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)  # Párhuzamossági limit beállítása
    tasks = [process_with_semaphore(semaphore, img) for img in images]
    
    await asyncio.gather(*tasks)  # A feladatok végrehajtása a megadott limit szerint

if __name__ == "__main__":
    asyncio.run(main())
