#Visual_RAG
## Multimodality Experiments

A collection of experiments for exploring multimodal retrieval and similarity methods.  
This repository hosts a Jupyter notebook demonstrating how to load, preprocess, and compare text and image embeddings using various libraries (NumPy, Pandas, Gensim, scikit-learn, Hugging Face Hub, etc.).

## 📂 Project Structure

### Multimodality_Experiments
-  Multimodality_Experiments.ipynb
-  README.md
-  requirements.txt


### MultiModal RAG Workflow

[ColPali](https://arxiv.org/abs/2407.01449) is a new multimodal retrieval system that seamlessly enables image retrieval.

By directly encoding image patches, it eliminates the need for optical character recognition (OCR), or image captioning to extract text from PDFs.

We will use `byaldi`, a library from [AnswerAI](https://www.answer.ai/), that makes it easier to work with an upgraded version of ColPali, called ColQwen2, to embed and retrieve images of our PDF documents.

Retrieved pages will then be passed into the Llama-3.2 90B Vision model served via a [Together AI](https://www.together.ai/) inference endpoint for it to answer questions.

