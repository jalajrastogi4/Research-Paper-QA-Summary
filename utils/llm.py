from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from core.config import settings
from core.logging import get_logger

logger = get_logger()

class LLM:
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.embeddings_model = settings.EMBEDDINGS_MODEL

    def get_llm(self, model_name: str = None, temperature: float = None) -> ChatOpenAI:
        try:
            model_name = model_name or self.model_name
            temperature = temperature or self.temperature
            llm = ChatOpenAI(model_name=model_name, 
                              temperature=temperature,
                              openai_api_key=settings.OPENAI_API_KEY)
            
            logger.info(f"LLM initialized with model: {model_name}, temperature: {temperature}")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}")

    
    def get_embeddings(self, model_name: str = None) -> OpenAIEmbeddings:
        try:
            model_name = model_name or self.embeddings_model
            embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=settings.OPENAI_API_KEY)
            logger.info(f"Embeddings initialized with model: {model_name}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise RuntimeError(f"Embeddings initialization failed: {e}")


llm_model = LLM()