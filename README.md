# Visual_RAG
## Multimodality Experiments

A collection of experiments for exploring multimodal retrieval and similarity methods.  
This repository hosts a Jupyter notebook demonstrating how to load, preprocess, and compare text and image embeddings using various libraries (NumPy, Pandas, Gensim, scikit-learn, Hugging Face Hub, etc.).

## 📂 Project Structure

### **Multimodality_Experiments**
-  Multimodality_Experiments.ipynb
-  README.md
-  requirements.txt


### **MultiModal RAG Operation Principal:**

[ColPali](https://arxiv.org/abs/2407.01449) is a new multimodal retrieval system that seamlessly enables image retrieval.

By directly encoding image patches, it eliminates the need for optical character recognition (OCR), or image captioning to extract text from PDFs.

We will use `byaldi`, a library from [AnswerAI](https://www.answer.ai/), that makes it easier to work with an upgraded version of ColPali, called ColQwen2, to embed and retrieve images of our PDF documents.

Retrieved pages will then be passed into the Llama-3.2 90B Vision model served via a [Together AI](https://www.together.ai/) inference endpoint for it to answer questions.

![ColPali vs Standard Retrieval Pipeline](images/Colpali_vs_Standard.png)

### **Pipeline Workflow:**
- #### Data Ingestion & Patch Encoding\
Each PDF page is sliced into visual patches that are fed directly into the ColQwen2 vision-LLM encoder, bypassing OCR and captioning. The encoder projects every patch into a dense embedding space in ≈ 0.4 s / page, forming a multi-vector index that preserves layout and visual cues.
- #### Multi-Vector Retrieval with byaldi\
A ColBERT-style MaxSim search over the patch index pulls the top-k relevant pages for any user query. Because queries can be text or image, byaldi embeds them in the same space.
- #### Reasoning via Llama-3.2 90B Vision\
Retrieved pages (images) and the original query are packed into a prompt and sent to Together AI’s hosted Llama-3.2 90B-Vision model. The model performs cross-page reasoning—summaries, Q&A, or citation grounding—while leveraging the visual context maintained in the patches.


