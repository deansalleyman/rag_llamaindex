# syntax=docker/dockerfile:1

FROM python:3.12-slim

# HF_HOME is set at BUILD time so the embedding model baked in below lands here,
# and at RUN time so the app finds it in the image instead of re-downloading.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Install CPU-only torch first so the heavy CUDA wheels are never pulled in,
# then the rest of the deps resolve against this already-present torch.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model weights into the image (~130 MB) so no download
# happens at runtime. Cached under HF_HOME=/opt/hf-cache in this image layer.
RUN python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY rag_pipeline.py .

# ./documents and ./chroma_db are provided as volumes at runtime (see compose).
ENTRYPOINT ["python", "rag_pipeline.py"]
