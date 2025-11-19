FROM python:3.8-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-tk \
    tk-dev \
    x11-apps \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data models ResultadosGuardados

EXPOSE 5000

# Usar X virtual framebuffer como alternativa
CMD ["xvfb-run", "-a", "python", "main.py"]