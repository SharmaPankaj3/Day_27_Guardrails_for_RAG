import os
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, DataType, Configure

# ---------- CONFIG ----------
WEAVIATE_API_KEY_PATH = r"D:\desktop\rag-guradrails-28_gurad_rail_with_rag.txt"
OPENAI_KEY_PATH = r"D:\desktop\Key_GEN_AI.txt"
TXT_PATH = r"D:\desktop\Day_27_Guardrails_for_RAG\Generative AI_doc.txt"
WEAVIATE_URL = "https://ampltuosc21b320qhhkfw.c0.asia-southeast1.gcp.weaviate.cloud"
COLLECTION_NAME = "DocumentTest"
BATCH_SIZE = 10
# ----------------------------

# Load keys (files closed properly)
with open(WEAVIATE_API_KEY_PATH, "r", encoding="utf-8") as f:
    WEAVIATE_API_KEY = f.read().strip()

with open(OPENAI_KEY_PATH, "r", encoding="utf-8") as f:
    OPENAI_API_KEY = f.read().strip()

# Set OpenAI key for text2vec-openai (if your class uses it)
os.environ["OPENAI_APIKEY"] = OPENAI_API_KEY

# Connect to Weaviate (v4 style)
auth = AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth
)

print("Connected:", client.is_ready())

# Create class/collection if not exists
if not client.collections.exists(COLLECTION_NAME):
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
else:
    print(f"Collection {COLLECTION_NAME} already exists")

collection = client.collections.get(COLLECTION_NAME)

# Read and split docs
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
with collection.batch.fixed_size(batch_size=10) as batch:
    for d in docs:
        batch.add_object(
            properties={"title": d["title"], "content": d["content"]}
        )
    failed = collection.batch.failed_objects
    if failed:
        print("Some objects failed:", failed)

print("Ingestion attempt done")

client.close()