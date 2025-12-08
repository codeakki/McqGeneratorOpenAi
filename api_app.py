import os
import json
import pickle
import traceback
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import PyPDF2
import io

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MCQ Generator API",
    description="API to upload documents and generate MCQs based on difficulty level (supports online/offline mode)",
    version="1.0.0"
)

# ==================== LLM CONFIGURATION ====================
# Set USE_OFFLINE=true in .env file to use Ollama (offline mode)
# Set USE_OFFLINE=false or leave unset to use OpenAI (online mode)

USE_OFFLINE = os.getenv('USE_OFFLINE', 'false').lower() == 'true'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')  # Default Ollama model

if True:
    # Offline mode using Ollama
    from langchain_community.llms import Ollama
    print(f"🔌 Running in OFFLINE mode with Ollama model: {OLLAMA_MODEL}")
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.5)
else:
    # Online mode using OpenAI
    from langchain_openai import ChatOpenAI
    from langchain_community.callbacks import get_openai_callback
    KEY = os.getenv('OPENAI_API_KEY')
    print(f"🌐 Running in ONLINE mode with OpenAI")
    llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)

# Directory to store pickle files
PICKLE_DIR = "pickle_files"
os.makedirs(PICKLE_DIR, exist_ok=True)

# Response JSON template for MCQ format
RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    }
}

# Request model for MCQ generation
class MCQGenerateRequest(BaseModel):
    difficulty_level: int = Field(..., ge=1, le=10, description="Difficulty level from 1 (easiest) to 10 (hardest)")
    subject: Optional[str] = Field(default="General", description="Subject for the MCQs")
    pickle_filename: Optional[str] = Field(default="document.pkl", description="Name of the pickle file to use")


def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded PDF or TXT file"""
    content = file.file.read()
    
    if file.filename.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(e)}")
    
    elif file.filename.endswith('.txt'):
        return content.decode('utf-8')
    
    else:
        raise HTTPException(status_code=400, detail="File format not supported. Please upload PDF or TXT file.")


def get_difficulty_tone(level: int) -> str:
    """Convert numeric difficulty level to descriptive tone"""
    tones = {
        1: "very simple and basic, suitable for beginners",
        2: "simple and easy to understand",
        3: "easy with some foundational concepts",
        4: "moderate with basic analytical thinking",
        5: "intermediate level requiring good understanding",
        6: "moderately challenging with deeper concepts",
        7: "challenging requiring analytical skills",
        8: "difficult with complex reasoning",
        9: "very difficult requiring expert knowledge",
        10: "extremely challenging, expert level only"
    }
    return tones.get(level, "intermediate")


def create_mcq_chain():
    """Create the MCQ generation chain"""
    template = """
    Text:{text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to format your response like RESPONSE_JSON below and use it as a guide ensure to return json format data only. \
    Ensure to make {number} MCQs
    ### RESPONSE_JSON
    {response_json}
    """
    
    quiz_generation_prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone", "response_json"],
        template=template
    )
    
    quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
    
    template2 = """
    You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
    if the quiz is not at per with the cognitive and analytical abilities of the students,\
    update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
    Quiz_MCQs:
    {quiz}

    Check from an expert English Writer of the above quiz:
    """
    
    quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=template2)
    review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)
    
    generate_evaluate_chain = SequentialChain(
        chains=[quiz_chain, review_chain],
        input_variables=["text", "number", "subject", "tone", "response_json"],
        output_variables=["quiz", "review"],
        verbose=True
    )
    
    return generate_evaluate_chain


def get_table_data(quiz_str: str) -> list:
    """Convert quiz string to table data format"""
    try:
        # Try to extract JSON from the response (handles cases where model adds extra text)
        json_start = quiz_str.find('{')
        json_end = quiz_str.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            quiz_str = quiz_str[json_start:json_end]
        
        quiz_dict = json.loads(quiz_str)
        quiz_table_data = []
        
        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " || ".join(
                [f"{option}-> {option_value}" for option, option_value in value["options"].items()]
            )
            correct = value["correct"]
            quiz_table_data.append({
                "question_no": key,
                "mcq": mcq,
                "choices": options,
                "correct": correct
            })
        
        return quiz_table_data
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return None


# ==================== API ENDPOINTS ====================

@app.get("/",
         summary="API Info",
         description="Get information about the API and current mode")
async def root():
    """Get API info and current mode"""
    return {
        "name": "MCQ Generator API",
        "version": "1.0.0",
        "mode": "offline (Ollama)" if USE_OFFLINE else "online (OpenAI)",
        "model": OLLAMA_MODEL if USE_OFFLINE else "gpt-3.5-turbo"
    }


@app.post("/upload-file/", 
          summary="Upload and convert file to pickle",
          description="Upload a PDF or TXT file to extract text and save as pickle file for later MCQ generation")
async def upload_file(
    file: UploadFile = File(...),
    filename: Optional[str] = None
):
    """
    API 1: Upload a file and convert it to a pickle file
    
    - **file**: PDF or TXT file to upload
    - **filename**: Optional custom name for the pickle file (without extension)
    """
    try:
        # Extract text from the uploaded file
        text = extract_text_from_file(file)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Could not extract any text from the file")
        
        # Generate pickle filename
        pickle_filename = filename if filename else file.filename.rsplit('.', 1)[0]
        pickle_path = os.path.join(PICKLE_DIR, f"{pickle_filename}.pkl")
        
        # Save text to pickle file
        data = {
            "original_filename": file.filename,
            "text": text,
            "text_length": len(text)
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        return JSONResponse(content={
            "success": True,
            "message": "File uploaded and converted to pickle successfully",
            "pickle_filename": f"{pickle_filename}.pkl",
            "original_filename": file.filename,
            "text_length": len(text),
            "text_preview": text[:500] + "..." if len(text) > 500 else text
        })
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/generate-mcqs/",
          summary="Generate MCQs from pickle file",
          description="Generate 10 MCQs based on difficulty level from a previously uploaded pickle file")
async def generate_mcqs(request: MCQGenerateRequest):
    """
    API 2: Generate 10 MCQs from pickle file based on difficulty level
    
    - **difficulty_level**: Difficulty from 1 (easiest) to 10 (hardest)
    - **subject**: Subject area for the MCQs (default: General)
    - **pickle_filename**: Name of the pickle file to use (default: document.pkl)
    """
    try:
        # Load text from pickle file
        pickle_path = os.path.join(PICKLE_DIR, request.pickle_filename)
        
        if not os.path.exists(pickle_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Pickle file '{request.pickle_filename}' not found. Please upload a file first."
            )
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        text = data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="No text found in the pickle file")
        
        # Get difficulty tone
        tone = get_difficulty_tone(request.difficulty_level)
        
        # Create MCQ chain and generate
        chain = create_mcq_chain()
        
        # Generate response JSON template for 10 MCQs
        response_template = {}
        for i in range(1, 11):
            response_template[str(i)] = {
                "no": str(i),
                "mcq": "multiple choice question",
                "options": {
                    "a": "choice here",
                    "b": "choice here",
                    "c": "choice here",
                    "d": "choice here"
                },
                "correct": "correct answer"
            }
        
        # Track token usage for OpenAI, skip for Ollama
        token_info = None
        
        if USE_OFFLINE:
            # Offline mode - no token tracking
            response = chain({
                "text": text,
                "number": 10,
                "subject": request.subject,
                "tone": tone,
                "response_json": json.dumps(response_template, indent=2)
            })
        else:
            # Online mode - track tokens
            from langchain_community.callbacks import get_openai_callback
            with get_openai_callback() as cb:
                response = chain({
                    "text": text,
                    "number": 10,
                    "subject": request.subject,
                    "tone": tone,
                    "response_json": json.dumps(response_template, indent=2)
                })
                token_info = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
        
        # Parse the quiz response
        quiz_data = response.get("quiz", None)
        review = response.get("review", None)
        
        result = {
            "success": True,
            "mode": "offline" if USE_OFFLINE else "online",
            "difficulty_level": request.difficulty_level,
            "difficulty_description": tone,
            "subject": request.subject,
        }
        
        if quiz_data:
            table_data = get_table_data(quiz_data)
            if table_data:
                result["mcqs"] = table_data
            else:
                result["raw_quiz"] = quiz_data
        
        result["review"] = review
        
        if token_info:
            result["tokens_used"] = token_info
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")


@app.get("/list-pickle-files/",
         summary="List available pickle files",
         description="List all pickle files available for MCQ generation")
async def list_pickle_files():
    """List all available pickle files"""
    try:
        files = []
        for filename in os.listdir(PICKLE_DIR):
            if filename.endswith('.pkl'):
                filepath = os.path.join(PICKLE_DIR, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                files.append({
                    "filename": filename,
                    "original_filename": data.get("original_filename", "Unknown"),
                    "text_length": data.get("text_length", 0)
                })
        
        return JSONResponse(content={
            "success": True,
            "files": files,
            "total_files": len(files)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.delete("/delete-pickle-file/{filename}",
            summary="Delete a pickle file",
            description="Delete a specific pickle file")
async def delete_pickle_file(filename: str):
    """Delete a specific pickle file"""
    try:
        filepath = os.path.join(PICKLE_DIR, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        os.remove(filepath)
        return JSONResponse(content={
            "success": True,
            "message": f"File '{filename}' deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.get("/health",
         summary="Health check",
         description="Check if the API is running")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "MCQ Generator API is running",
        "mode": "offline (Ollama)" if USE_OFFLINE else "online (OpenAI)"
    }


# Run with: uvicorn api_app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
