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
from rank_bm25 import BM25Okapi

MOCK_MODE = os.getenv("MOCK_MODE", "False").lower() == "true"
SIMILARITY_THRESHOLD = 0.80 # Only check if embeddings are 80%+ similar
CONFIRMATION_THRESHOLD = 0.95 # Auto-confirm only if 95%+ similar (nearly identical text)

def check_semantic_duplicate(new_text, candidate_text):
    """
    Uses LLM to verify if two texts mean the same thing.
    Returns True ONLY if they are asking for the exact same specific wish.
    """
    if MOCK_MODE: return False
    
    # Quick string match
    if new_text.strip().lower() == candidate_text.strip().lower():
        return True

    from ai_service import get_gemini_model
    try:
        model = get_gemini_model()
        prompt = f"""
You are a strict duplicate detector. Your task is to determine if two wishes are asking for the EXACT SAME THING.

IMPORTANT RULES:
- "I want a bike" and "I want a car" are DIFFERENT - they ask for different objects
- "I want a laptop" and "I want a computer" might be SAME if they mean the exact same model/specs
- Different nouns = DIFFERENT wishes (even if in same category like vehicles, gadgets, etc)
- Only answer YES if they're asking for the same specific thing

Answer with ONLY "true" or "false" (no explanations).

Wish A: "{new_text}"
Wish B: "{candidate_text}"

Are these the EXACT SAME wish?
        """
        print("before model")
        response = model.generate_content(prompt)
        answer = response.text.strip().lower()
        # Only return True if the response is exactly "true"
        return answer == "true"
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

def calculate_bm25_scores(query_text, documents_texts):
    """
    Calculate BM25 scores for a query against multiple documents.
    Returns a list of scores normalized to [0, 1].
    """
    if not documents_texts or not query_text:
        return []
    
    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents_texts]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_docs)
    
    # Query tokenized
    query_tokens = query_text.lower().split()
    
    # Score all documents
    scores = bm25.get_scores(query_tokens)
    
    # Normalize scores to [0, 1] using sigmoid-like normalization
    # BM25 scores can go beyond 1, so we normalize them
    max_score = max(scores) if len(scores) > 0 else 1.0
    normalized_scores = [min(score / max(max_score, 1.0), 1.0) for score in scores]
    
    return normalized_scores

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
        existing_problems = list(database.problems.find({"embedding": {"$exists": True}}, {"embedding": 1, "original_text": 1, "_id": 1, "analysis": 1}))
        
        best_match = None
        max_combined_score = -1.0
        embedding_score = 0.0
        bm25_score = 0.0
        
        if existing_problems:
            # Calculate BM25 scores for all existing problems
            existing_texts = [p.get("original_text", "") for p in existing_problems]
            bm25_scores = calculate_bm25_scores(submission.description, existing_texts)
            
            # Find best match using combined scoring
            for idx, problem in enumerate(existing_problems):
                emb_score = cosine_similarity(embedding, problem.get("embedding"))
                bm25_sc = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
                
                # Combine scores: 60% embedding + 40% BM25
                combined_score = (0.6 * emb_score) + (0.4 * bm25_sc)
                
                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_match = problem
                    embedding_score = emb_score
                    bm25_score = bm25_sc
        
        # 2. If Similarity > Threshold, treat as duplicate
        if max_combined_score > SIMILARITY_THRESHOLD and best_match:
            is_dupe = False
            
            # Case A: Extremely high combined similarity (nearly identical text)
            if max_combined_score > CONFIRMATION_THRESHOLD:
                is_dupe = True
                print(f"Auto-Duplicate (Embedding: {embedding_score:.2f}, BM25: {bm25_score:.2f}, Combined: {max_combined_score:.2f})")
            
            # Case B: High match but requires LLM verification
            else:
                print(f"Potential Duplicate (Embedding: {embedding_score:.2f}, BM25: {bm25_score:.2f}, Combined: {max_combined_score:.2f}). Verifying with LLM...")
                is_dupe = check_semantic_duplicate(submission.description, best_match.get("original_text"))
                
            if is_dupe:
                print(f"Confirmed Duplicate! '{submission.description}' == '{best_match.get('original_text')}'")
                database.problems.update_one(
                    {"_id": best_match["_id"]},
                    {"$inc": {"vote_count": 1}}
                )
                existing_analysis = best_match.get("analysis", {})
                return {
                    "id": str(best_match["_id"]),
                    "status": "upvoted_existing",
                    "message": "Similar wish already exists! We folded your wish into it.",
                    "similarity_score": max_combined_score,
                    "embedding_score": embedding_score,
                    "bm25_score": bm25_score,
                    "resources": existing_analysis.get("resources", [])
                }
            else:
                print(f"False Positive detected. Score: {max_combined_score} but LLM said NO.")
                
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
