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
            data_ollama = {"taskType": "describe", "modelType": "groq", "modelName": "llama-3.2-11b-vision-preview"}
            response_ollama = await client.post(API_URL, files=files, data=data_ollama)
            response_ollama.raise_for_status()
            print(f"✅ Ollama response received for {image_path}")

            result_ollama = response_ollama.json().get("extracted_text", "")

            # Ollama llava-llama-3-8b feldolgozás
            data_ollama_1 = {"taskType": "describe", "modelType": "groq", "modelName": "llama-3.2-90b-vision-preview"}
            response_ollama_1 = await client.post(API_URL, files=files, data=data_ollama_1)
            response_ollama_1.raise_for_status()
            print(f"✅ Ollama response received for {image_path}")

            result_ollama_1 = response_ollama_1.json().get("extracted_text", "")

            # Kombinálás OpenAI GPT-4o segítségével
            # Art
            painting_prompt = (
                "Merge the following three image descriptions into a single, detailed, and well-structured analysis. "
                "Ensure the final description is comprehensive, factually accurate, and free of any hallucinations. "
                "Prioritize factual correctness, logical consistency, and coherence while preserving all relevant details.\n\n"
                f"**Description 1 (GPT-4o):** {result_openai}\n\n"
                f"**Description 2 (Ollama llava-llama-3-11b):** {result_ollama}\n\n"
                f"**Description 3 (Ollama llava-llama-3-90b):** {result_ollama_1}\n\n"
                "Your response should be in Hungarian. If there are any discrepancies between the descriptions, "
                "resolve them by using the most accurate, logically consistent, and contextually appropriate details. "
                "Structure the response into distinct sections covering style, composition, color palette, lighting, "
                "brushstroke techniques, mood, symbolism, historical context, and overall interpretation."
            )
            # Historycal photos
            prompt = (
                "Combine the following three historical photograph descriptions into a single, structured, and detailed analysis. "
                "The final description must be comprehensive, factually accurate, and free of any hallucinations. "
                "Prioritize factual correctness, logical consistency, and coherence while ensuring that all relevant details are preserved.\n\n"
                
                f"**Description 1 (GPT-4o - Primary Source):** {result_openai}\n\n"
                f"**Description 2 (Ollama Llava3.2 - Secondary Source):** {result_ollama}\n\n"
                f"**Description 3 (Ollama llava-llama-3-8b - Supplementary Source):** {result_ollama_1}\n\n"
                
                "Your response should be in Chinese. When discrepancies arise between the descriptions, "
                "resolve them by prioritizing the most factually accurate, logically consistent, and contextually appropriate details. "
                "If subjective or stylistic elements from a source enhance the description without contradicting facts, integrate them where appropriate.\n\n"
                
                "Structure your response into **clearly defined sections**, covering:\n"
                "- **Time Period and Location** (historical and geographical context)\n"
                "- **Composition and Lighting** (framing, depth of field, and use of light and shadow)\n"
                "- **Photographic Techniques** (camera positioning, exposure, and stylistic choices)\n"
                "- **Depicted Subjects** (people, clothing, objects, and their arrangement)\n"
                "- **Social and Historical Context** (cultural significance, events, or historical backdrop)\n"
                "- **Interpretation and Overall Significance** (artistic, documentary, or symbolic meaning of the photograph)\n\n"
                
                "Ensure that the final analysis maintains a neutral, objective tone, focusing on the photograph’s historical and artistic value."
                "Make sure the counting of objects. Crosscheck! if only one model count or models count differently, then do not include the number into the output"
            )

            data_generate = {"query": prompt, "modelType": "openai", "modelName": "gpt-4o"}
            response_combined = await client.post(API_GENERATE, data=data_generate)
            response_combined.raise_for_status()
            print(f"✅ Combined text response received for {image_path}")

            final_text = response_combined.json().get("text", "")

            # Eredmény mentése fájlba
            output_path = os.path.join(OUTPUT_DIR, f"{Path(image_path).stem}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                # f.write(f"1.\n {result_openai}\n")
                # f.write(f"2.\n {result_ollama}\n")
                # f.write(f"3.\n {result_ollama_1}\n")
                f.write(f"Final:\n {final_text}\n")

            print(f"✅ Processed successfully: {image_path} -> {output_path}")
    except httpx.TimeoutException:
        print(f"❌ Timeout error processing {image_path}")
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error {e.response.status_code} processing {image_path}: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error processing {image_path}: {e}")


CONCURRENT_TASKS = 5  # Maximum párhuzamos feldolgozások száma

async def process_with_semaphore(semaphore, image_path):
    """Egy kép feldolgozása a megadott szemináriummal."""
    async with semaphore:  # Limitáljuk az egyidejű feldolgozásokat
        await process_image(image_path)

async def main():
    """Az összes kép feldolgozása az IMAGE_DIR könyvtárban, korlátozott párhuzamossággal."""
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff"}
    images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS]

    if not images:
        print("❌ No images found in the directory.")
        return

    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)  # Párhuzamossági limit beállítása
    tasks = [process_with_semaphore(semaphore, img) for img in images]
    
    await asyncio.gather(*tasks)  # A feladatok végrehajtása a megadott limit szerint

if __name__ == "__main__":
    asyncio.run(main())
