# Gebruik de NVIDIA CUDA base image
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Stel de working directory in de container in
WORKDIR /app

# Installeer systeemafhankelijkheden: ffmpeg voor video, git voor kloon, python3-pip, libcudnn voor GPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3-pip \
    libcudnn8 libcudnn8-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installeer Python-afhankelijkheden
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Kopieer het script naar de container
COPY createSrt.py .

# Geef aan dat /data de plek is waar de video's komen
VOLUME /data

# Zorg ervoor dat de uitvoerbuffer van Python direct wordt geflusht
ENV PYTHONUNBUFFERED=1

# Definieer de entrypoint van de container. Argumenten die aan 'docker run' worden meegegeven,
# worden hierachter geplakt.
ENTRYPOINT ["python3", "createSrt.py"]

# Standaard commando als er geen argumenten worden meegegeven.
# Dit zal de standaardtaal "nl" gebruiken als argument voor het script.
CMD ["--language", "nl"]