import os
import math
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.auth import AuthApiKey

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Langchain + OpenAI imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.chains import RetrievalQA

# helpers
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError, Field
import numpy as np

# ---------- CONFIG ----------
WEAVIATE_API_KEY_PATH = r"D:\desktop\rag-guradrails-28_gurad_rail_with_rag.txt"
OPENAI_KEY_PATH = r"D:\desktop\Key_GEN_AI.txt"
TXT_PATH = r"D:\desktop\Day_27_Guardrails_for_RAG\Generative AI_doc.txt"
WEAVIATE_URL = "https://ampltuosc21b320qhhkfw.c0.asia-southeast1.gcp.weaviate.cloud"
COLLECTION_NAME = "DocumentTest"
BATCH_SIZE = 10

# Retrieval / grounding thresholds
SIMILARITY_THRESHOLD =  0.65
MIN_CONTEXT_TO_PASS = 1
GROUNDING_PROMPT_CONFIDENCE_THRESHOLD = 0.7

# Safety keywords
SAFETY_BLOCKLIST = [
    "bomb", "detonate", "explode", "how to hack", "password", "brute force", "illegal", "poison",
    "assassinate", "harm", "weapon", "terrorist"
                    ]

# Load keys (files closed properly)
with open(WEAVIATE_API_KEY_PATH, "r", encoding="utf-8") as f:
    WEAVIATE_API_KEY = f.read().strip()

with open(OPENAI_KEY_PATH, "r", encoding="utf-8") as f:
    OPENAI_API_KEY = f.read().strip()

# Set OpenAI key for text2vec-openai (if your class uses it)
os.environ["OPENAI_APIKEY"] = OPENAI_API_KEY

headers = {
    "X-OpenAI-Api-Key": OPENAI_API_KEY,
}

# Connect to Weaviate (v4 style)
auth = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth,
    headers=headers
)

print("Connected:", client.is_ready())

# path to your plain text file (the content you asked me to generate)
#  TXT_PATH = r"D:\desktop\Day_27_Guardrails_for_RAG\Generative AI_doc.txt"

# thresholds
TOP_K = 5
# Ensure collection exists( we won't re ingest here)
# Create class/collection if not exists
# ---------- Ensure collection exists and ingest documents ----------
# Create class/collection if not exists
if client.collections.exists(COLLECTION_NAME):
    client.collections.delete(COLLECTION_NAME)
    print(f"Deleted old collection {COLLECTION_NAME}")

client.collections.create(
    name=COLLECTION_NAME,
    description="RAG documents collection",
    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ],
    vector_config=Configure.Vectors.text2vec_openai(model="text-embedding-3-small")
)
print(f"Collection {COLLECTION_NAME} created")
collection = client.collections.get(COLLECTION_NAME)


# Read and split docs from TXT
with open(TXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

parts = [p.strip() for p in text.split("###") if p.strip()]
docs = []
for p in parts:
    lines = p.splitlines()
    title = lines[0].strip()
    content = "\n".join(lines[1:]).strip()
    if content:
        docs.append({"title": title, "content": content})

print(f"Preparing to ingest {len(docs)} documents")

# Batch ingestion
with collection.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
    for d in docs:
        batch.add_object(properties={"title": d["title"], "content": d["content"]})
    failed = collection.batch.failed_objects
    if failed:
        print("Some objects failed:", failed)

print("Ingestion complete")

print(client.is_ready())

# LLM & Embeddings

llm =ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",api_key=OPENAI_API_KEY)

# Vectorstore wrapper
vectorstore = WeaviateVectorStore(client, index_name=COLLECTION_NAME,text_key="content", embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":8})

# Output schema(pydantic)
class RAGResponse(BaseModel):
    answer: str = Field(..., description="Final answer presented to user")
    grounded: bool = Field(..., description="Whether answer is supported by retrieved contexts ")
    sources: List[str] = Field(..., description="List of source snippets or ids used")
    confidence: Optional[float] = Field(None, description="Optional grounding confidence 0-1")
    note: Optional[str] = Field(None, description="Fallback note if not grounded or refused")

# Utility Function
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # safe cosine
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_query_unsafe(query: str) -> Optional[str]:
    q_lower = query.lower()
    for kw in SAFETY_BLOCKLIST:
        if kw in q_lower:
            return kw
    return None

def embed_text(texts:List[str]) -> List[np.ndarray]:
    # embeddings.embed_documents returns list of lists; convert to numpy arrays
     embs = embeddings.embed_documents(texts)
     arrs = [np.array(e, dtype=float) for e in embs]
     return arrs

def retrieve_and_filter(query: str, top_k: int = 8, similarity_threshold: float = SIMILARITY_THRESHOLD):
    """
    1) Use retriever to fetch candidate docs(k).
    2) Embed query and docs, compute cosine similarity.
    3) Keep only docs above threshold (return list of dicts: { text, metadata, score})
    """
    # get candidate docs from retriever
    docs = retriever.invoke(query)
    if not docs:
        return[]

   # prepare texts

    texts = [d.page_content for d in docs]
    doc_embs = embed_text(texts)
    query_emb = np.array(embeddings.embed_query(query), dtype=float)

    filtered = []
    for doc, emb in zip(docs, doc_embs):
        score = cosine_similarity(query_emb, emb)
        if score >= similarity_threshold:
            filtered.append({"text": doc.page_content, "meta": getattr(doc, "metadata", {}), "score": score})

        # sort by descending score
    filtered.sort(key=lambda x: x["score"], reverse=True)
    return filtered

def llm_generate_answer(query: str, contexts: List[Dict[str,Any]]) -> str:
    """
    Robust LLM call to generate an answer constrained to provided contexts.
    Returns the answer text (string). On failure returns a descriptive error string
    starting with "LLM_ERROR:" so caller can detect failures if needed.
    """
    contexts_text = "\n\n".join([f"Context {i+1}:\n{c.get('text', c.get('page_content', str(c)))}"
                                 for i, c in enumerate(contexts)])
    system_msg = (
        "You are an assistant that answers user queries using ONLY the provided contexts. "
        "If the answer is not supported by the contexts, reply exactly: 'NOT_ENOUGH_CONTEXT'. "
        "Keep the answer concise (1-3 sentences) and list sources if asked."
    )
    user_msg = f"User question: {query}\n\nContexts (do not use outside knowledge):\n{contexts_text}\n\nAnswer now:"

    try:
        # call the llm (keep your invoke call style)
        resp = llm.invoke([{"role": "system", "content": system_msg},
                           {"role": "user", "content": user_msg}])
    except Exception as e:
        return f"LLM_ERROR: invoke failed: {str(e)}"

    # Extract text content robustly from resp
    ans_text = None
    try:
        # 1) attribute .content
        if hasattr(resp, "content"):
            ans_text = resp.content
        # 2) mapping-like with keys
        elif isinstance(resp, dict):
            # common keys: 'result', 'content', 'choices'
            if "content" in resp and isinstance(resp["content"], str):
                ans_text = resp["content"]
            elif "result" in resp and isinstance(resp["result"], str):
                ans_text = resp["result"]
            elif "choices" in resp and isinstance(resp["choices"], (list, tuple)) and resp["choices"]:
                ch = resp["choices"][0]
                # try several nested shapes
                if isinstance(ch, dict):
                    # openai style: choices[0]['message']['content']
                    if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                        ans_text = ch["message"]["content"]
                    elif "text" in ch:
                        ans_text = ch["text"]
                else:
                    ans_text = str(ch)
        # 3) try attribute 'result' or string form
        elif hasattr(resp, "result"):
            ans_text = getattr(resp, "result")
        else:
            ans_text = str(resp)
    except Exception:
        ans_text = None

    # final safety: ensure string and not None
    if ans_text is None:
        try:
            ans_text = str(resp)
        except Exception as e:
            return f"LLM_ERROR: cannot extract answer content: {str(e)}"

    # normalize to string
    if not isinstance(ans_text, str):
        try:
            ans_text = str(ans_text)
        except Exception:
            return "LLM_ERROR: answer content not a string"

    return ans_text.strip()


def verify_grounding_with_llm(answer: str, contexts:List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ask LLM (verifier) whether 'answer' is supported by 'contexts'.
    Returns: {"grounded": bool, "explanation": str}
    Robust extraction and safe fallbacks.
    """
    contexts_text = "\n\n".join([f"Context {i+1}:\n{c.get('text', c.get('page_content', str(c)))}"
                                 for i, c in enumerate(contexts)])
    prompt = (
        "You are a verifier. Given an ANSWER and supporting CONTEXTS, decide whether the ANSWER"
        " is directly supported by the CONTEXTS. Answer in a short JSON with fields: "
        '"grounded" (true/false) and "explanation" (a short sentence). '
        "Do not use outside knowledge; judge only on the contexts.\n\n"
        f"Answer: {answer}\n\nContexts:\n{contexts_text}\n\nRespond now:"
    )

    try:
        resp = llm.invoke([{"role": "user", "content": prompt}])
    except Exception as e:
        return {"grounded": False, "explanation": f"Verifier invoke failed: {str(e)}"}

    # extract text from resp robustly (reuse logic)
    text = None
    try:
        if hasattr(resp, "content"):
            text = resp.content
        elif isinstance(resp, dict):
            if "content" in resp and isinstance(resp["content"], str):
                text = resp["content"]
            elif "result" in resp and isinstance(resp["result"], str):
                text = resp["result"]
            elif "choices" in resp and resp["choices"]:
                ch = resp["choices"][0]
                if isinstance(ch, dict):
                    if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                        text = ch["message"]["content"]
                    elif "text" in ch:
                        text = ch["text"]
                else:
                    text = str(ch)
        elif hasattr(resp, "result"):
            text = getattr(resp, "result")
        else:
            text = str(resp)
    except Exception:
        text = None

    if not text:
        try:
            text = str(resp)
        except Exception as e:
            return {"grounded": False, "explanation": f"Verifier response parse failed: {str(e)}"}

    # Try to parse a JSON blob in the response
    try:
        import json, re
        json_block = re.search(r"\{.*\}", text, re.DOTALL)
        if json_block:
            parsed = json.loads(json_block.group(0))
            grounded = bool(parsed.get("grounded", False))
            explanation = parsed.get("explanation", "") or str(parsed)
            return {"grounded": grounded, "explanation": explanation}
    except Exception:
        # fallthrough to keyword heuristics
        pass

    # Heuristic fallback: look for yes/true or no/false
    t = text.lower()
    if "true" in t or "yes" in t or "grounded" in t and ("true" in t or "yes" in t):
        return {"grounded": True, "explanation": text}
    if "false" in t or "no" in t:
        return {"grounded": False, "explanation": text}

    # default: not grounded, return full verifier text as explanation
    return {"grounded": False, "explanation": text}


# Main RAG with guardrails
def answer_query_with_guardrails(user_query: str) -> Dict[str, Any]:
    """
    Guarded RAG response pipeline:
      1) safety check
      2) retrieve/filter contexts
      3) if not enough context -> conservative NOT_ENOUGH_CONTEXT
      4) generate answer constrained to contexts
      5) if model indicates NOT_ENOUGH_CONTEXT -> conservative fallback
      6) verify grounding
      7) return validated RAGResponse
    """
    # 1) Safety check (simple blocklist)
    blocked_kw = is_query_unsafe(user_query)
    if blocked_kw:
        return RAGResponse(
            answer="I am sorry - I can't help with that request.",
            grounded=False,
            sources=[],
            confidence=0.0,
            note=f"Blocked due to unsafe keyword: {blocked_kw}"
        ).dict()

    # 2) Retrieve and filter contexts
    try:
        filtered_contexts = retrieve_and_filter(user_query, top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD)
    except Exception as e:
        # retrieval failed for some reason
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note=f"Retrieval failed: {str(e)}"
        ).dict()

    if not filtered_contexts or len(filtered_contexts) < MIN_CONTEXT_TO_PASS:
        # not enough high quality contexts - fallback
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note="Not enough relevant retrieved context to answer reliably."
        ).dict()

    # 3) Generate answer constrained to contexts
    try:
        generated = llm_generate_answer(user_query, filtered_contexts)
        if generated is None:
            generated = ""
        generated = generated.strip()
    except Exception as e:
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note=f"Generation failed: {str(e)}"
        ).dict()

    # 4) If model explicitly indicates lack of context -> fallback
    if generated.upper().startswith("NOT_ENOUGH_CONTEXT"):
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note="Model indicated lack of context."
        ).dict()

    # 5) Verify grounding via the verifier LLM (or other verifier)
    try:
        verification = verify_grounding_with_llm(generated, filtered_contexts)
    except Exception as e:
        # if verifier fails, be conservative
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note=f"Verifier failed: {str(e)}"
        ).dict()

    grounded = bool(verification.get("grounded", False))
    explanation = verification.get("explanation", "")

    if not grounded:
        return RAGResponse(
            answer="NOT_ENOUGH_CONTEXT",
            grounded=False,
            sources=[],
            confidence=0.0,
            note=f"Answer not supported by retrieved contexts. Verifier: {explanation}"
        ).dict()

    # 6) Prepare sources list (short snippets)
    sources = []
    try:
        for c in filtered_contexts[:5]:
            txt = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
            if not txt:
                # try other keys
                txt = c.get("page_content") if isinstance(c, dict) else getattr(c, "page_content", "")
            snippet = (txt[:400].replace("\n", " ")) if txt else ""
            sources.append(snippet)
    except Exception:
        # fallback empty list
        sources = []

    # 7) Build final validated object
    try:
        resp_obj = RAGResponse(
            answer=generated,
            grounded=True,
            sources=sources,
            confidence=1.0,
            note="Supported by retrieved contexts."
        )
        return resp_obj.dict()
    except ValidationError as e:
        # schema validation failed - conservative fallback
        return {
            "answer": "NOT_ENOUGH_CONTEXT",
            "grounded": False,
            "sources": [],
            "confidence": 0.0,
            "note": f"Schema validation failed: {str(e)}"
        }

    return resp_obj.dict()

# Example usage

if __name__ == "__main__":
    # quick interactive REPL

    print("Day 27 RAG Guardrails demo. Type a question (or 'exit').")

    while True:
        q = input("\nQuestion ").strip()
        if q.lower() in ("exit", "quit"):
            break
        out = answer_query_with_guardrails(q)
        print("\n--- Response ---")
        print(out)

    # close weaviate connection at the end.
    client.close()
