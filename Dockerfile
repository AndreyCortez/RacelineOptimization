# Use uma imagem oficial do Python como base
FROM python:3.9

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

RUN pip install --upgrade setuptools

# Atualiza o Cython para a versão mais recente
# RUN pip install cython==0.29.19

# Copia o arquivo requirements.txt para o diretório de trabalho
COPY vendor/laptime-simulation/requirements.txt . 
COPY vendor/laptime-simulation/ app/

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt --no-build-isolation

# Copia o conteúdo atual do diretório para o diretório de trabalho no contêiner
COPY . .

# Comando para executar o aplicativo quando o contêiner for iniciado
CMD ["python", "app.py"]
