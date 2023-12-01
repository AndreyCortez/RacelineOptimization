# Use uma imagem oficial do Python como base
FROM python:3.7

RUN apt-get update \
    && apt-get install -y python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

RUN pip install --upgrade setuptools

# Atualiza o Cython para a versão mais recente
# RUN pip install cython==0.29.19

# Copia o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt . 

# Instala as dependências do Python
RUN pip install --upgrade pip
RUN pip install --upgrade cython
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt --no-build-isolation

# Copia o conteúdo atual do diretório para o diretório de trabalho no contêiner
COPY . .

# Comando para executar o aplicativo quando o contêiner for iniciado
CMD ["python", "genetic.py"]
