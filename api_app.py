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


# Request model for Chatbot
class ChatbotRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the document content")
    level: int = Field(default=5, ge=1, le=10, description="Explanation level: 1 = most detailed/simple, 10 = brief/advanced")
    pickle_filename: Optional[str] = Field(default="document.pkl", description="Name of the pickle file to use")


# Response format template for chatbot
CHATBOT_RESPONSE_FORMAT = {
    "is_related": True,  # Boolean: whether question is related to content
    "topic": "Topic name from the chapter",
    "answer": "The detailed answer here",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "summary": "A brief one-line summary"
}


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


def get_explanation_style(level: int) -> str:
    """Convert level to explanation style - lower level = more detailed explanation"""
    styles = {
        1: "Explain in the most detailed and simple way possible. Use very easy words, give multiple examples, break down every concept step by step like explaining to a child. Be thorough and patient.",
        2: "Explain in great detail using simple language. Provide several examples and analogies. Break complex ideas into smaller parts.",
        3: "Explain thoroughly with clear examples. Use simple vocabulary and provide step-by-step explanations.",
        4: "Explain clearly with good examples. Use accessible language and provide helpful context.",
        5: "Explain in a balanced way with moderate detail. Include relevant examples where helpful.",
        6: "Explain concisely but clearly. Include key examples only when necessary.",
        7: "Explain efficiently, focusing on main points. Assume some background knowledge.",
        8: "Explain briefly and directly. Focus on key concepts without extensive elaboration.",
        9: "Explain in a compact, advanced manner. Use technical terms freely.",
        10: "Explain very briefly and concisely. Assume expert-level understanding. Be direct and to the point."
    }
    return styles.get(level, styles[5])


def create_chatbot_chain():
    """Create the chatbot chain for Q&A based on document content"""
    template = """
    You are a helpful assistant that answers questions ONLY based on the provided text content.
    
    IMPORTANT RULES:
    1. ONLY answer if the question is related to the content provided below
    2. If the question is NOT related to the content or cannot be answered from the content, set is_related to false
    3. Do NOT make up information or use external knowledge
    4. Base your answer STRICTLY on the provided text
    5. ALWAYS respond in the exact JSON format specified below
    
    TEXT CONTENT:
    {text}
    
    EXPLANATION STYLE:
    {explanation_style}
    
    USER QUESTION:
    {question}
    
    RESPOND IN THIS EXACT JSON FORMAT ONLY (no other text before or after):
    {{
        "is_related": true or false,
        "topic": "The specific topic from the chapter this question relates to (or 'N/A' if not related)",
        "answer": "Your detailed answer here following the explanation style (or 'Sorry, please ask a question related to the chapter content.' if not related)",
        "key_points": ["Key point 1", "Key point 2", "Key point 3"],
        "summary": "A brief one-line summary of the answer"
    }}
    
    JSON RESPONSE:
    """
    
    chatbot_prompt = PromptTemplate(
        input_variables=["text", "explanation_style", "question"],
        template=template
    )
    
    chatbot_chain = LLMChain(llm=llm, prompt=chatbot_prompt, output_key="answer", verbose=True)
    
    return chatbot_chain


def create_mcq_chain():
    """Create the MCQ generation chain"""
    template = """
    Text:{text}
    You are an expert MCQ maker. Given the above text, it is your job to \
    create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
    
    IMPORTANT RULES:
    1. Each question must have EXACTLY ONE correct answer (NO multiple selection questions)
    2. Each question must have exactly 4 options: a, b, c, d
    3. Only ONE option should be correct
    4. The "correct" field must contain ONLY the letter of the correct option (a, b, c, or d)
    5. Make sure questions are not repeated
    6. All questions must be based on the provided text
    7. Return ONLY valid JSON format, no other text
    
    Make sure to format your response like RESPONSE_JSON below and use it as a guide ensure to return json format data only. \
    Ensure to make {number} MCQs with SINGLE correct answer each.
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
        
        # Generate response JSON template for 10 MCQs (single correct answer only)
        response_template = {}
        for i in range(1, 11):
            response_template[str(i)] = {
                "no": str(i),
                "mcq": "Write the question here",
                "options": {
                    "a": "First option",
                    "b": "Second option",
                    "c": "Third option",
                    "d": "Fourth option"
                },
                "correct": "a"  # ONLY ONE letter (a, b, c, or d) - single correct answer
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


@app.post("/chat/",
          summary="Chat with document content",
          description="Ask questions about the uploaded document. Lower level = more detailed explanation.")
async def chat_with_document(request: ChatbotRequest):
    """
    API 3: Chatbot - Ask questions about the document content
    
    - **question**: Your question about the document
    - **level**: Explanation level (1-10). Lower = more detailed, Higher = more brief
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
        
        # Get explanation style based on level
        explanation_style = get_explanation_style(request.level)
        
        # Create chatbot chain and get answer
        chain = create_chatbot_chain()
        
        # Track token usage for OpenAI, skip for Ollama
        token_info = None
        
        if USE_OFFLINE:
            # Offline mode - no token tracking
            response = chain({
                "text": text,
                "explanation_style": explanation_style,
                "question": request.question
            })
        else:
            # Online mode - track tokens
            from langchain_community.callbacks import get_openai_callback
            with get_openai_callback() as cb:
                response = chain({
                    "text": text,
                    "explanation_style": explanation_style,
                    "question": request.question
                })
                token_info = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
        
        raw_answer = response.get("answer", "").strip()
        
        # Try to parse the structured JSON response
        parsed_response = None
        try:
            # Extract JSON from the response
            json_start = raw_answer.find('{')
            json_end = raw_answer.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = raw_answer[json_start:json_end]
                parsed_response = json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        result = {
            "success": True,
            "mode": "offline" if USE_OFFLINE else "online",
            "question": request.question,
            "level": request.level,
            "level_description": f"Level {request.level}: {'Very detailed' if request.level <= 3 else 'Moderate detail' if request.level <= 6 else 'Brief/Advanced'}",
        }
        
        if parsed_response:
            # Structured response
            result["response"] = {
                "is_related": parsed_response.get("is_related", True),
                "topic": parsed_response.get("topic", "General"),
                "answer": parsed_response.get("answer", ""),
                "key_points": parsed_response.get("key_points", []),
                "summary": parsed_response.get("summary", "")
            }
        else:
            # Fallback to raw answer if parsing fails
            result["response"] = {
                "is_related": True,
                "topic": "General",
                "answer": raw_answer,
                "key_points": [],
                "summary": ""
            }
        
        if token_info:
            result["tokens_used"] = token_info
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


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
