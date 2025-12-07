from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from utils.prompts import qa_prompt
from utils.llm import llm_model
from core.config import settings
from core.logging import get_logger

logger = get_logger()


def answer_question(vector_store, question: str):
    try:
        llm = llm_model.get_llm()
        prompt = qa_prompt()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        logger.info(f"Answering question: {question}")
        answer = retrieval_chain.invoke({"question": question})
        logger.info(f"Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        raise RuntimeError(f"Failed to answer question: {e}")