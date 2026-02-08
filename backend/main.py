import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from db import db
from ai_service import evaluate_problem, generate_audio_summary, get_text_embedding
from mock_data import MOCK_OPPORTUNITIES
from datetime import datetime
import base64
import math
import uuid

MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"
SIMILARITY_THRESHOLD = 0.85 # Preliminary filter
CONFIRMATION_THRESHOLD = 0.96 # Auto-confirm if extremely high

def check_semantic_duplicate(new_text, candidate_text):
    """
    Uses LLM to verify if two texts mean the same thing.
    Returns True if they are semantic duplicates.
    """
    if MOCK_MODE: return False
    
    # Quick string match
    if new_text.strip().lower() == candidate_text.strip().lower():
        return True

    from ai_service import get_gemini_model
    try:
        model = get_gemini_model()
        prompt = f"""
        Task: Determine if these two user wishes are fundamentally the same request.
        
        Wish A: "{new_text}"
        Wish B: "{candidate_text}"
        
        Answer strictly "YES" or "NO".
        YES = They differ only in phrasing but ask for the exact same core solution.
        NO = They ask for different things, or one is a sub-category of the other.
        """
        response = model.generate_content(prompt)
        answer = response.text.upper().strip()
        return "YES" in answer
    except Exception as e:
        print(f"LLM Duplicate Check Failed: {e}")
        return False

def cosine_similarity(v1, v2):
    "Compute cosine similarity between two vectors."
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not MOCK_MODE:
        db.connect()
    yield
    # Shutdown
    if not MOCK_MODE:
        db.close()

app = FastAPI(lifespan=lifespan)

class ProblemSubmission(BaseModel):
    description: str

@app.post("/submit")
async def submit_problem(submission: ProblemSubmission):
    embedding = get_text_embedding(submission.description)
    database = None
    
    if not MOCK_MODE:
        database = db.get_db()
        # --- IDEMPOTENCY CHECK START ---
        # 1. Fetch all problems with embeddings
        # Optimization: In prod, use Vector Search (Atlas) or FAISS. 
        # Here we do a linear scan which is fine for <10k items.
        existing_problems = database.problems.find({"embedding": {"$exists": True}}, {"embedding": 1, "original_text": 1})
        
        best_match = None
        max_score = -1.0
        
        for problem in existing_problems:
            score = cosine_similarity(embedding, problem.get("embedding"))
            if score > max_score:
                max_score = score
                best_match = problem
        
        # 2. If Similarity > Threshold, treat as duplicate
        if max_score > SIMILARITY_THRESHOLD and best_match:
            is_dupe = False
            
            # Case A: Extremely high vector similarity (nearly identical text)
            if max_score > CONFIRMATION_THRESHOLD:
                is_dupe = True
                print(f"Auto-Duplicate (Score: {max_score})")
            
            # Case B: High embedding match but requires LLM verification
            else:
                print(f"Potential Duplicate (Score: {max_score}). Verifying with LLM...")
                is_dupe = check_semantic_duplicate(submission.description, best_match.get("original_text"))
                
            if is_dupe:
                print(f"Confirmed Duplicate! '{submission.description}' == '{best_match.get('original_text')}'")
                database.problems.update_one(
                    {"_id": best_match["_id"]},
                    {"$inc": {"vote_count": 1}}
                )
                return {
                    "id": str(best_match["_id"]),
                    "status": "upvoted_existing",
                    "message": "Similar wish already exists! We folded your wish into it.",
                    "similarity_score": max_score
                }
            else:
                print(f"False Positive detected. Score: {max_score} but LLM said NO.")
                
        # --- IDEMPOTENCY CHECK END ---

    # 3. AI Analysis (only if new)
    analysis = evaluate_problem(submission.description)
    
    # 4. Store in DB
    document = {
        "original_text": submission.description,
        "embedding": embedding,
        "vote_count": 1,
        "analysis": analysis,
        "created_at": datetime.utcnow(),
        "audio_b64": None
    }
    
    # 5. Generate Audio only if it's a "Solvable" opportunity
    if analysis.get("is_public"):
        audio_bytes = generate_audio_summary(analysis.get("summary", "New opportunity available"))
        if audio_bytes:
            document["audio_b64"] = base64.b64encode(audio_bytes).decode('utf-8')

    if MOCK_MODE:
        # Simulate ID generation and return immediately
        return {"id": str(uuid.uuid4()), "status": "submitted (mock)", "analysis": analysis}

    result = database.problems.insert_one(document)
    
    return {"id": str(result.inserted_id), "status": "submitted", "analysis": analysis}

@app.get("/opportunities")
async def get_opportunities():
    if MOCK_MODE:
        return MOCK_OPPORTUNITIES

    database = db.get_db()
    # Fetch only public/solvable problems
    # Sort by vote_count descending (popular first), then newest
    cursor = database.problems.find({"analysis.is_public": True}).sort([("vote_count", -1), ("created_at", -1)])
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Ensure older records have a visual default
        if "vote_count" not in doc:
            doc["vote_count"] = 1
        results.append(doc)
        
    return results

@app.get("/")
def home():
    return {"message": "API is running"}
