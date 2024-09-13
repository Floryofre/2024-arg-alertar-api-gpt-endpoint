from dotenv import load_dotenv
import openai
import os
import json
from fastapi import APIRouter, FastAPI

# Crear el enrutador para "users"
chat_router = APIRouter()
# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_SECRET_KEY')

# Nombre del archivo JSONL
filename = "./data/context_model_prompts.jsonl"

context_model_prompts = list()

# Escribir los datos en el archivo JSONL
with open(filename, 'r') as jsonl:
    for line in jsonl:
        entry = json.loads(line.strip())
        context_model_prompts.append(entry)

model_params = {
    'model' : 'o1-mini',
    'top_p' :  0.4,
    'max_tokens':1024,
    'messages': context_model_prompts
}

@chat_router.post('/chat/{user_prompt}')
async def generate_response(user_prompt: str):
    """
    Genera una respuesta a partir de un prompt del usuario.
    
    Args:
        user_prompt (str): El prompt proporcionado por el usuario.
    
    Returns:
        dict: Respuesta generada por el modelo.
    """
    global context_model_prompts
    user_query = {{"role": "user", "content": user_prompt}}
    try:
        context_model_prompts.append(user_query)
        response = openai.Completion.create(
            model=model_params['model'],
            max_tokens=model_params['max_tokens'],
            n=1,
            top_p = model_params['top_p'],
            stop=None,
            temperature=0.3,
            messages = messages,
        )
        model_response = response.choices[0].message.content
        # Guarda la respuesta del modelo de manera temporal en la memoria
        # para que sea permanente habria que generar un id unico por usuario en futuras versiones
        context_model_prompts.append({"role": "assistant", "content": model_response})
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la respuesta: {str(e)}")