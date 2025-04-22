# LLM CONFIG
dimensions: int = 1536
embedding_provider: str = "openai"
embedding_model: str = "text-embedding-3-small"
llm_provider: str = "openai" # Existing provider setting
llm_interface: str = "openai" # NEW: Feature flag for internal implementation ('openai' or 'pydantic')
llm_model: str = "gpt-4o"
openai_api_key: Optional[str] = None 