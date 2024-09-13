#libreria para variables de entorno
from dotenv import load_dotenv
#Libreria para validar el tipo de datos de entrada
from pydantic import BaseModel
# libreria de OpenAI
import openai
# Librerias para manejar directorios y archivos no nativos de python
import os
import json
# librerias de fastapi
from fastapi import APIRouter, FastAPI

class UserPrompt(BaseModel):
    user_prompt: str

# Router del apartado /chat ubicado dentro de www.mapabullying.com.ar
chat_router = APIRouter()

# Carga de variables de entorno almacenadas en .env 
load_dotenv()
# Obtencion de la secret_key y guardado dentro de metodo de openai.api_key
openai.api_key = os.getenv('OPENAI_API_SECRET_KEY')

# instancia de fastapi 
app = FastAPI()

# archivo jsonl con pre entramiento/contexto de entrada para el modelo instanciado de OpenAI
filename = "./data/context_model_prompts.jsonl"

# Lista donde se volcaran cada una de las entradas del jsonl
context_model_prompts = list()

# Volcado de entradas del jsonl en la lista "context_model_prompts"
with open(filename, 'r') as jsonl:
    for line in jsonl:
        entry = json.loads(line.strip())
        context_model_prompts.append(entry)

# Parametros a pasar a la instancia del modelo de OpenAI
model_params = {
    'model' : 'o1-mini', #modelo elegido
    'top_p' :  0.35, #Presicion/soltura del modelo - eligira las palabras para las respuestas que tengan un 65% o mas de probabilidad
    'max_tokens':1024, #Cantidad maxima de tokens por respuesta tanto de entrada como de salida
    'messages': context_model_prompts #Pre-entrenamiento/contexto del modelo
}

@chat_router.post('/chat/')
async def generate_response(prompt: UserPrompt):
    """
    Genera una respuesta a partir de un prompt del usuario.
    
    Args:
        prompt (str): El prompt proporcionado por el usuario que se validara con el onjeto heredado de BaseModel.
    
    Returns:
        dict: Respuesta generada por el modelo.
    """
    global context_model_prompts
    user_prompt = prompt.user_prompt
    user_query = {"role": "user", "content": user_prompt}
    try:
        #Agregado de la consulta del usuario en la ultima entrada del jsonl
        context_model_prompts.append(user_query)
        #Creacion del response o respuesta del modelo en base a la pregunta o prompt del usuario
        response = openai.chat.completions.create(
            model=model_params['model'],
            max_tokens=model_params['max_tokens'],
            n=1,
            top_p = model_params['top_p'],
            stop=None,
            temperature=0.3,
            messages = context_model_prompts,
        )
        # Seleccion y guardado del objeto 'content' que contiene la respuesta del modelo
        model_response = response.choices[0].message.content
        # Guarda la respuesta del modelo de manera temporal en la memoria
        # para que sea permanente habria que generar un id unico por usuario en futuras versiones
        context_model_prompts.append({"role": "assistant", "content": model_response})
        #retorna solo la respuesta del asistente
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la respuesta: {str(e)}")