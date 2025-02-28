import os
import base64
from fastapi import APIRouter, File, UploadFile
from services.text_extraction import (
    extract_text_from_pdf_bytes,
    extract_text_from_image_bytes,
    extract_text_from_docx_bytes,
    extract_text_from_pptx_bytes,
    extract_text_from_txt_bytes,
    extract_text_from_url,
    extract_youtube_transcript
)
from google import genai

# Load the Gemini API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Ensure the API key is found
if not api_key:
    raise ValueError("API key for Gemini not found. Set the GEMINI_API_KEY environment variable.")

router = APIRouter()

# Function to extract text from images using Gemini
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    client = genai.Client(api_key=api_key)
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[img_base64, "Extract text from this image"]
    )
    return response.text if response.text else "No text extracted from the image."

# Route to handle PDF text extraction
@router.post("/extract_pdf/")
async def extract_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    text = extract_text_from_pdf_bytes(pdf_bytes)
    return {"source": file.filename, "type": "pdf", "extracted_text": text}

# Route to handle image (OCR) text extraction
@router.post("/extract_image/")
async def extract_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    text = extract_text_from_image_bytes(image_bytes)
    return {"source": file.filename, "type": "image", "extracted_text": text}

# Route to handle Word (.docx) text extraction
@router.post("/extract_docx/")
async def extract_docx(file: UploadFile = File(...)):
    docx_bytes = await file.read()
    text = extract_text_from_docx_bytes(docx_bytes)
    return {"source": file.filename, "type": "docx", "extracted_text": text}

# Route to handle PowerPoint (.pptx) text extraction
@router.post("/extract_pptx/")
async def extract_pptx(file: UploadFile = File(...)):
    pptx_bytes = await file.read()
    text = extract_text_from_pptx_bytes(pptx_bytes)
    return {"source": file.filename, "type": "pptx", "extracted_text": text}

# Route to handle plain text (.txt) extraction
@router.post("/extract_txt/")
async def extract_txt(file: UploadFile = File(...)):
    txt_bytes = await file.read()
    text = extract_text_from_txt_bytes(txt_bytes)
    return {"source": file.filename, "type": "txt", "extracted_text": text}

# Route to extract text from a URL
@router.get("/extract_url/")
async def extract_url(url: str):
    text = extract_text_from_url(url)
    return {"source": url, "type": "url", "extracted_text": text}

# Route to extract YouTube transcript
@router.get("/extract_youtube_transcript/")
async def extract_youtube_transcript_route(video_url: str):
    transcript = extract_youtube_transcript(video_url)
    return {"source": video_url, "type": "youtube", "extracted_text": transcript}



