"""Index and retrieve images using Byaldi's RAGMultiModalModel."""

import os
from pathlib import Path
from typing import List, Union, Optional

from byaldi import RAGMultiModalModel
from huggingface_hub import login
from dotenv import load_dotenv

class ImageRetriever:
    """
    Indexes and retrieves chart images using a multimodal retriever.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v0.1",
        index_name: str = "image_index",
        api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initializes the ImageRetriever with the specified model and index configurations.

        Args:
            model_name (str): Name of the pretrained multimodal model.
            index_name (str): Name of the index to create or update.
            api_key (str, optional): API key for authentication with Together AI.
            hf_token (str, optional): Token for authentication with Hugging Face Hub.
        """
        # Load environment variables if any
        load_dotenv()

        # Set API keys if provided
        if api_key:
            os.environ["TOGETHER_API_KEY"] = api_key
        if hf_token:
            login(hf_token)

        self.model = RAGMultiModalModel.from_pretrained(model_name)
        self.index_name = index_name

    def index_images(self, input_path: Union[str, Path], overwrite: bool = True):
        """
        Indexes images from the specified input path.

        Args:
            input_path (Union[str, Path]): Path to the directory containing images.
            overwrite (bool): Whether to overwrite the existing index.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path {input_path} does not exist.")

        print(f"Indexing images from {input_path} into index '{self.index_name}'...")
        self.model.index(
            input_path=input_path,
            index_name=self.index_name,
            store_collection_with_index=True,  # Stores base64 images along with the vectors
            overwrite=overwrite,
        )
        print(f"Indexing completed for '{self.index_name}'.")

    def retrieve_images(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Retrieves images based on a query.

        Args:
            query (str): The query string.
            top_k (int): Number of top results to retrieve.

        Returns:
            List[dict]: List of retrieved images with metadata.
        """
        results = self.model.search(
            query=query,
            index_name=self.index_name,
            top_k=top_k,
        )
        return results
