# BrunoChat
A chatbot designed to serve as a conversational knowledge interface for the Computer Science Department. 

## Data Preparation
Refer to [BrunoChat Data Pipelines](https://github.com/MadaniKK/2270-crawler-test). 

## Installation
### 1. Activate Virtual Environment
```shell
python -m venv brunochat-env
source brunochat-env/bin/activate
```

### 2. Install Dependencies
To install packages
```shell
pip install -r requirements.txt 
```

To export packages
```shell
pip freeze > requirements.txt
```

### 3. Prepare Credentials
```shell
echo OPENAI_API_KEY={openai_key}   >> .env
echo QDRANT_API_KEY={qdrant_key}   >> .env
echo QDRANT_URL={server_url}       >> .env
```

### 4. Launch
```shell
streamlit run src/ui.py
```