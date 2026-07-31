# Use a lightweight python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Optimize PyTorch & system memory usage for 512MB RAM limits
ENV MALLOC_ARENA_MAX=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Pre-download ensemble JIT weights during Docker build to avoid download RAM overhead at runtime
RUN mkdir -p convnext_results vit_results && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='chrisnguyenx/ConvNeXt-P3', filename='convnext_inference.pt', local_dir='convnext_results')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='chrisnguyenx/DeiT-ViT-P3', filename='vit_inference.pt', local_dir='vit_results')"

# Copy code
COPY preprocessing.py .
COPY main_api.py .

# Expose port
EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
