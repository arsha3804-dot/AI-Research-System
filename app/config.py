from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "groq"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    fallback_api_key: str = ""
    fallback_base_url: str = "https://openrouter.ai/api/v1"
    fallback_model: str = "openrouter/free"

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./data/chroma_db"
    docs_dir: str = "./data/docs"
    mcp_server_script: str = "./mcp_server/research_tools_server.py"
    max_iterations: int = 5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
