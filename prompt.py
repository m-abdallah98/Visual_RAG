from llama_index.core import PromptTemplate
import os

## Defines the RAG/Pure LLM Prompt
def get_prompt(language, retrieval=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../task_prompt.txt")
    with open(file_path, "r") as f:
        task_prompt = f.read()

    # In case of RAG
    if retrieval:
        rag_context_prompt = """
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        {image_str}
        """
        # Adjust the image section based on whether images are available
        rag_question_prompt = f"Given the context information and any images provided, answer the query while keeping your response only in {language} and avoid any {language} spelling mistakes.\n"

        rag_prompt = task_prompt + rag_context_prompt + rag_question_prompt
        RAG_PROMPT = PromptTemplate(rag_prompt)
        PROMPT = RAG_PROMPT

        # In case of Pure LLM
    else:
        norag_context_prompt = """
        Query: {query_str}
        """
        norag_question_prompt = f"Respond to this query in {language}.\n"
        norag_prompt = task_prompt + norag_context_prompt + norag_question_prompt
        BASIC_PROMPT = PromptTemplate(norag_prompt)
        PROMPT = BASIC_PROMPT
    return PROMPT
