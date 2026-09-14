FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    BACKEND_URL=http://127.0.0.1:8000

WORKDIR $HOME/app

COPY --chown=user requirements-deploy.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 & python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true"]
