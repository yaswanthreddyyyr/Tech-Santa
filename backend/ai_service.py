import os
import json
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    logger.warning("google-generativeai package not found.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"

# Global model instance
_model_instance = None

def get_gemini_model():
    global _model_instance
    if _model_instance:
        return _model_instance
        
    if not HAS_GENAI or not GEMINI_API_KEY:
        logger.error("GenAI not available or API Config missing")
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Preferred models in order - Updated for Feb 2026
        preferred_models = ["models/gemini-2.0-flash", "models/gemini-2.5-flash", "models/gemini-1.5-flash"]
        valid_model_name = "models/gemini-2.0-flash" # Default fallback
        
        # In a real production scenario, we might want to validate the model once
        # possibly list_models() here but catch errors to avoid blocking startup
        # For speed, we will default to flash if not configured otherwise
        
        logger.info(f"Initializing AI Model: {valid_model_name}")
        _model_instance = genai.GenerativeModel(valid_model_name)
        return _model_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize GenAI: {e}")
        return None

def get_text_embedding(text: str) -> list[float]:
    """Generates an embedding vector for the given text."""
    if MOCK_MODE:
        # Return a mock 768-dim vector
        return [0.1] * 768
        
    if not HAS_GENAI or not GEMINI_API_KEY:
        logger.warning("Cannot generate embedding: GenAI not configured")
        return []

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Wish Embedding"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return []

def evaluate_problem(problem_text: str):
    """
    Uses Gemini to classify the problem.
    """
    if MOCK_MODE:
        return {
            "category": "Solvable",
            "industry": "Healthcare",
            "summary": "Mock summary: " + problem_text[:50] + "...",
            "guidance": "Use Python + Pandas + Streamlit",
            "reasoning": "This is a mock response because MOCK_MODE is enabled.",
            "is_public": True
        }

    # List of models to try in order of preference
    # We use multiple aliases to increase chances of hitting a working/quota-available model
    models_to_try = [
        "models/gemini-2.0-flash", 
        "models/gemini-2.5-flash", 
        "models/gemini-1.5-flash", 
        "models/gemini-flash-latest"
    ]
    
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(f"Attempting AI analysis with model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            prompt = f"""
            Analyze the following user input: "{problem_text}".

            **Crucial Instruction: Infer Intent deeply.**
            - Users often express problems as sentiments ("I hate standing in line" -> Problem: Queues are inefficient).
            - Users often express wishes as affection for non-existent things ("I love flying bikes" -> Problem: Personal aerial transport doesn't exist yet). Deduce the missing product.
            - Do NOT dismiss an input just because it says "I love..." or "I hate...". Translate it into a technical or product challenge.
            
            Return a strictly valid JSON object (no markdown formatting) with the following keys:
            1. "category": One of ["Solvable", "Unsolvable", "Spam", "Existing Solution"].
            2. "industry": A high-level industry tag (e.g., "Healthcare", "Automotive", "FinTech", "Retail", "Education", "General").
            3. "summary": A 1-sentence summary of the *inferred* problem or desire.
            4. "guidance": If solvable, a short tech stack recommendation. If existing, name the solution.
            5. "reasoning": A short explanation of why you chose this category and how you inferred the problem.
            6. "is_public": Boolean. True ONLY if the category is "Solvable".
            """
            
            response = model.generate_content(prompt)
            
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(text_response)
            
            # If successful, return immediately
            logger.info(f"Success with model: {model_name}")
            return data
            
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            last_error = e
            # Continue to next model in the list
            continue
            
    # If all models fail
    logger.error(f"All AI models failed. Last error: {last_error}")
    return {
        "category": "Unprocessed", 
        "industry": "Unknown",
        "summary": "AI could not process", 
        "guidance": "Check manually", 
        "reasoning": str(last_error),
        "is_public": False
    }

def generate_audio_summary(text: str):
    """
    Uses ElevenLabs to generate audio. Returns audio bytes.
    """
    if MOCK_MODE:
        # Return empty bytes or a dummy indicator in mock mode
        print("Mock mode: Skipping audio generation")
        return None

    if not ELEVENLABS_API_KEY:
        return None
        
    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM" # Default "Rachel" voice
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content
    return None
