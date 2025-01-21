from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import os
import logging
import base64
import json
import httpx  # Aszinkron HTTP kliens
from typing import List, Optional
from dotenv import load_dotenv
from model_manager import ModelManager  # ModelManager importálása

# Környezeti változók betöltése
load_dotenv()

# Logger beállítása
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI alkalmazás inicializálása
app = FastAPI()

# CORS middleware hozzáadása az API-hoz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model_options = {
    "ollama": ["llama3.2-vision", "llava", "nsheth/llava-llama-3-8b-v1_1-int4"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "groq"  : ["llama-3.2-11b-vision-preview","llama-3.2-90b-vision-preview","Llama-3.3-70B-Instruct"]
}

# Jinja2 sablonok beállítása
templates = Jinja2Templates(directory="templates")

# ModelManager inicializálása az Ollama és OpenAI modellek kezelésére
def get_model_manager():
    model_type = os.getenv("MODEL_TYPE", "ollama")
    model_name = os.getenv("MODEL_NAME", "llama3.2-vision")
    return ModelManager(model_type=model_type, model_name=model_name)

model_manager = get_model_manager()

@app.post("/generate")
async def generate(
    query: str = Form("text") ,
    modelType: str = Form("ollama"),
    modelName: str = Form("llama3.2-vision")
):
    # Külső ModelManager osztály process_image metódusának meghívása
    extracted_text = await model_manager.generate_complete(
        messages=[{"role": "user", "content": query}],
        model_type=modelType,
        model_name=modelName
    )
    return JSONResponse(content={"text": extracted_text})
     

@app.post("/upload_ocr/")
async def upload_and_process_image(
    file: UploadFile = File(...),
    taskType: str = Form("describe"),
    modelType: str = Form("ollama"),
    modelName: str = Form("llama3.2-vision")
):
    try:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Kép base64 kódolása
        base64_image = encode_image_to_base64(temp_path)
        os.remove(temp_path)
        
        # Üzenet összeállítása OCR vagy kép leírás számára
        """
        # Read watermeter
        (
            "Act as a water meter reading assistant. Analyze the provided image and:\n"
            "1. Identify the main numeric reading on the meter (cubic meters, m³) and transcribe it accurately.\n"
            "3. If any numbers are unclear or partially obscured, indicate this with [unclear] in your transcription.\n"
            "4. Do not include any extra commentary; only provide the meter reading in a structured format as follows:\n"
            "{\n"
            '  "main_reading_m3": "XXXX",\n'
            "}."
        )
        """
        prompt = (
            "Act as an OCR assistant. Analyze the provided image and:\n"
            "1. Recognize all visible text in the image as accurately as possible.\n"
            "2. Maintain the original structure and formatting of the text.\n"
            "3. If any words or phrases are unclear, indicate this with [unclear] in your transcription.\n"
            "Provide only the transcription without any additional comments."
            if taskType == "ocr" else
             # analyse photo
            "Act as a historical photography analysis assistant. Describe the photograph in a structured and detailed manner, step by step. Start by identifying the general time period, style, and possible origin based on clothing, objects, and setting. Then, analyze the composition, lighting, depth of field, and any notable photographic techniques. Consider the subject matter, including the people, their attire, expressions, and possible social or historical context. Discuss any symbolic or cultural elements, as well as potential interpretations of the image. If the photograph has historical or documentary significance, provide relevant background information. If it is a staged or portrait photograph, focus on the posed arrangement, the photographer's intent, and any notable features. Conclude with a summary of its historical impact and unique characteristics."
            #"Act as an art analysis assistant. Describe the painting in a structured and detailed manner, step by step. Start by identifying the general style and artistic movement it belongs to. Then, analyze the composition, color palette, lighting, brushstroke techniques, and the overall mood. Consider the subject matter, including depicted figures, objects, and their interactions. Discuss any symbolic or emotional elements, as well as possible interpretations. If the painting has historical or cultural significance, provide relevant context. If the painting is abstract or modern, focus on its use of form, texture, and the emotions it evokes. Conclude with a summary of its artistic impact and unique characteristics."
        )
        
        # Külső ModelManager osztály process_image metódusának meghívása
        extracted_text = await model_manager.process_image(
            messages=[{"role": "user", "content": prompt, "images": [base64_image]}],
            model_type=modelType,
            model_name=modelName
        )
        
        if not extracted_text:
            raise HTTPException(status_code=500, detail="OCR failed to process the image.")
        
        return JSONResponse(content={"filename": file.filename, "extracted_text": extracted_text})
    except Exception as e:
        logger.error(f"Upload OCR error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing file.")

def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.get("/")
async def home(request: Request):
    # TemplateResponse létrehozása
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "model_options": model_options
    })
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
