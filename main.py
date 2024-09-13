from fastapi import FastAPI, HTTPException
from routers import chat  # Importar los enrutadores

app = FastAPI()

# Incluir los enrutadores en la aplicación principal
app.include_router(chat.chat_router)