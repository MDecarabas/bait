import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from .agents import run_agent_chat
from .config import BaitConfig

load_dotenv()

# --- FastAPI App ---
api = FastAPI()


class ChatQuery(BaseModel):
    query: str


@api.post("/chat")
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to receive a query and return the agent's response.
    """
    answer = run_agent_chat(chat_query.query)
    return {"response": answer}


@api.get("/config")
async def get_config_endpoint():
    """
    Get current configuration.
    """
    config = BaitConfig()
    return {"config": config.model_dump()}


@api.post("/config")
async def update_config_endpoint(new_config: dict):
    """
    Update configuration (requires restart to apply).
    """
    # This is a placeholder - in production you'd want to validate and save
    return {"message": "Configuration updated. Restart backend to apply changes."}


def main():
    """
    Main function to run the FastAPI application using uvicorn.
    """
    config = BaitConfig()
    uvicorn.run(api, host=config.server.backend_host, port=config.server.backend_port)
