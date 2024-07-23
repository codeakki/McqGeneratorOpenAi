AUTOMATED MCQ GENERATOR USING LANGCHAIN AND OPENAI-API

# How to run?

### STEPS:

Clone the repository

```bash
Project repo: https://github.com/codeakki/McqGeneratorOpenAi.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.9 -y
```

```bash
conda activate venv
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:

```

## Sample

![OpenAI Logo](https://github.com/codeakki/TextToSQLUsingGENAI/blob/main/image.png)

![Exapmple2](https://github.com/codeakki/TextToSQLUsingGENAI/blob/main/image2.png)

## HuggingFace

https://huggingface.co/spaces/codeakki/TextToSQLUsingGENAI/tree/main

### Techstack Used:

- Python
- OpenApi
- langchain
- Streamlit
- PyPDF2
