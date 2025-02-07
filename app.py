from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
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
app = FastAPI(
    docs_url=None,  
    redoc_url=None,
    openapi_url=None,
    redirect_slashes=True,
    root_path="",
    servers=[{"url": "/", "description": "Local server"}]  # Kikényszerített URL
)
# Explicit keep-alive header
@app.middleware("http")
async def add_keep_alive_header(request, call_next):
    response = await call_next(request)
    response.headers["Connection"] = "keep-alive"
    return response
    
# CORS middleware hozzáadása az API-hoz
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Engedélyezzük az összes domain-t
    allow_credentials=True,
    allow_methods=["*"],  # Engedélyezett minden HTTP metódus
    allow_headers=["*"]
)

# HTTPS Redirect Middleware 
# Távoli elérés esetén kell!!
# Lokális esetén nem!!!!
app.add_middleware(HTTPSRedirectMiddleware)

# Trusted Host Middleware
app.add_middleware(TrustedHostMiddleware)

# Kényszerítsd az OpenAI API hívásokat közvetlen netkapcsolatra
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


model_options = {
    "ollama": ["llama3.2-vision", "llava-llama3","bakllava","deepseek-r1","mistral-small:24b"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "deepseek"  : ["deepseek-chat","deepseek-reasoner"],
    "groq"  : ["llama-3.2-11b-vision-preview","llama-3.2-90b-vision-preview","llama3-8b-8192"]
}

# Jinja2 sablonok beállítása
templates = Jinja2Templates(directory="templates")

# ModelManager inicializálása az Ollama és OpenAI modellek kezelésére
def get_model_manager():
    model_type = os.getenv("MODEL_TYPE", "ollama")
    model_name = os.getenv("MODEL_NAME", "llama3.2-vision")
    return ModelManager(model_type=model_type, model_name=model_name)

model_manager = get_model_manager()


@app.post("/generate", include_in_schema=False)
@app.post("/generate/", include_in_schema=False)
async def generate(
    text: str = Form("text"),
    modelType: str = Form("ollama"),
    modelName: str = Form("llama3.2-vision")
):
    async def generate_response_stream():
        async for chunk in model_manager.generate_stream(
            model_type=modelType,
            model_name=modelName,
            messages=[{"role": "user", "content": text}]
        ):
            # Ha a chunk byte-típusú, akkor dekódoljuk UTF-8-ra
            if isinstance(chunk, bytes):
                try:
                    chunk = chunk.decode("utf-8")  # Átalakítjuk JSON-serializable formátumba
                    logger.info(chunk)
                except UnicodeDecodeError:
                    logger.error(f"Nem sikerült dekódolni a választ: {chunk}", exc_info=True)
                    continue  # Ha nem dekódolható, kihagyjuk

            yield f"data: {json.dumps(chunk)}\n\n"

    #return StreamingResponse(generate_response_stream(), media_type="text/event-stream")
    return StreamingResponse(generate_response_stream(), media_type="text/event-stream", headers={
        "X-Accel-Buffering": "no",  # Engedélyezi az azonnali küldést
        "Cache-Control": "no-cache",  # Kikapcsolja a cache-elést
        "Connection": "keep-alive"  # Megakadályozza a kapcsolat megszakadását
    })
@app.post("/generate_full")
async def generate_full(    
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
   
   
@app.post("/upload_ocr")
async def upload_and_process_image(
    file: UploadFile = File(...),
    text: str = Form("text") ,
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
        if not text:
            text = "\n"
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
            "Provide only the transcription without any additional comments.\n"
            f"{text}"
            if taskType == "ocr" else
                # analyse photo
               """ Act as an advanced historical photography analysis assistant. Provide a structured and detailed analysis of any given black-and-white photograph while ensuring precision and avoiding assumptions. Follow these structured steps:"

            1. Identify the time period and photographic style
            Estimate the historical time period based on clothing, objects, and setting.
            Describe the likely photographic process (e.g., daguerreotype, gelatin silver print, collodion).
            Mention if the image exhibits signs of an early or modern photographic technique (e.g., visible grain, exposure quality).
            
            2. Describe the setting and environmental elements
            Identify whether the image was taken in an urban, rural, or natural environment.
            If buildings are present, describe their architectural style and possible time period.
            If a natural landscape is visible, describe elements like trees, water, mountains, or other background features.
            If the location appears identifiable, describe the elements that support this conclusion.
            
            3. Analyze the people in the scene
            Describe their clothing, posture, and possible social status.
            Identify accessories such as hats, gloves, parasols, uniforms, or other notable attire.
            If they interact with objects, animals, or other people, describe the nature of the interaction.
            Do not assume actions, emotions, or relationships unless clearly visible.
            
            4. List and analyze visible objects and animals
            Identify objects present in the image and their possible function (e.g., furniture, vehicles, street signs).
            If animals are included, describe their placement and role in the scene.
            Do not add objects that are not clearly visible in the image.
            
            5. Analyze composition, lighting, and depth of field
            Describe the positioning of subjects and focal points within the image.
            Identify lighting conditions based on shadows and highlights.
            Analyze the depth of field and sharpness—what is in focus, and what is blurred?
            If applicable, note if the photograph captures motion or has a staged appearance.
            
            6. Avoid speculation or unnecessary assumptions
            Do not introduce details that are not visually present in the image.
            If unsure about the number of people, objects, or animals, use approximate language or indicate uncertainty.
            Do not infer specific locations, time periods, or cultural practices unless clear evidence supports them.
            
            7. Provide a historical and cultural interpretation
            If the image suggests a known historical or cultural practice, describe it while clarifying that it is inferred from historical context.
            Discuss how the photograph reflects social norms, fashion, or daily life of its time.
            If the image appears staged or documentary in nature, comment on its possible intent and significance.
            Deliver a structured, detailed, and evidence-based analysis, ensuring maximum accuracy while maintaining a neutral and objective approach.
            """
            f"{text}"
        )


        logger.info(f"Prompt: {prompt}")        
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        root_path="",
        forwarded_allow_ips="*",
        proxy_headers=True,
        access_log=True,  # Logolja a kéréseket a hibakereséshez
        timeout_keep_alive=300,  # A kapcsolat nyitva tartása hosszú stream esetén
        log_level="debug"
    )

