from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import List, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

app = FastAPI(title="Women's Support Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    language: str
    query: str
    chat_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    tips: List[str]
    chat_history: List[ChatMessage]

def get_response(language: str, query: str) -> tuple[str, List[str]]:
    """Generate friendly response and safety tips"""
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        prompt = f"""You are a friendly, caring person helping women with their concerns.
        Language: {language}
        Query: {query}
        
        Provide:
        1. A brief, caring response (2-3 sentences) as if talking to a friend
        2. Three short safety tips if relevant (if not, return empty list)
        
        Format the response as: [response] | [tip1] | [tip2] | [tip3]"""
        
        result = model.generate_content(prompt)
        parts = result.text.split('|')
        
        response = parts[0].strip()
        tips = [tip.strip() for tip in parts[1:] if tip.strip()]
        
        return response, tips
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the Women's Support Assistant"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    valid_languages = ["hindi", "english", "punjabi", "bengali", "marathi", "gujarati", "tamil", "telugu"]
    if request.language.lower() not in valid_languages:
        raise HTTPException(status_code=400, detail="Invalid language selection")

    try:
        response, tips = get_response(request.language, request.query)
        
        new_chat_history = list(request.chat_history)
        new_chat_history.extend([
            ChatMessage(role="user", content=request.query),
            ChatMessage(role="assistant", content=response)
        ])
        
        return ChatResponse(
            response=response,
            tips=tips,
            chat_history=new_chat_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
def get_available_languages():
    return {"languages": ["hindi", "english", "punjabi", "bengali", "marathi", "gujarati", "tamil", "telugu"]}