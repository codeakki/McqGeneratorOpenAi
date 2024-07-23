## AUTOMATED MCQ GENERATOR USING LANGCHAIN AND OPENAI-API



The "Automated MCQ Generator Using LangChain and OpenAI API" project aims to create a system that automatically generates multiple-choice questions (MCQs) using the LangChain library in conjunction with the OpenAI API. This system leverages advanced natural language processing techniques to analyze and understand textual content, then formulates relevant and accurate MCQs to aid in educational assessments and training programs. The automation of this process saves time and effort for educators and trainers, ensuring a high-quality and consistent question generation process.

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

![OpenAI Logo](https://github.com/codeakki/McqGeneratorOpenAi/blob/master/image.png)

![Exapmple2](https://github.com/codeakki/McqGeneratorOpenAi/blob/master/image2.png)



### Techstack Used:

- Python
- OpenApi
- langchain
- Streamlit
- PyPDF2
