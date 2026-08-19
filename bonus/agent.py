import time
from uuid import uuid4
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from feast import FeatureStore
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

class HybridMemoryAgent:
    def __init__(self, qdrant_client: QdrantClient = None, feast_repo_path: str = None):
        self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        self.qdrant = qdrant_client or QdrantClient(":memory:")
        if not self.qdrant.collection_exists("episodic_memory"):
            self.qdrant.create_collection(
                collection_name="episodic_memory",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
        repo_path = feast_repo_path or str(ROOT / "app" / "feast_repo")
        try:
            self.fs = FeatureStore(repo_path=repo_path)
        except Exception:
            self.fs = None # For demo purposes when Feast isn't fully initialized

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        vector = next(self.embedder.embed([text])).tolist()
        point_id = str(uuid4())
        
        self.qdrant.upsert(
            collection_name="episodic_memory",
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "text": text,
                        "timestamp": time.time()
                    }
                )
            ]
        )

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features -> return assembled context."""
        # 1. Get user profile + recent activity from Feast online store
        profile_context = ""
        recent_context = ""
        
        if self.fs:
            try:
                features = self.fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour"
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                
                speed = features.get("reading_speed_wpm", ["Unknown"])[0]
                topic = features.get("topic_affinity", ["Unknown"])[0]
                queries_1h = features.get("queries_last_hour", [0])[0]
                
                profile_context = f"User likes '{topic}' and reads at {speed} wpm."
                recent_context = f"Recent activity: {queries_1h} queries in the last hour."
            except Exception as e:
                profile_context = "Profile data unavailable."
                recent_context = "Activity data unavailable."
        else:
            profile_context = "Profile data unavailable (Feast not configured)."
            recent_context = "Activity data unavailable (Feast not configured)."

        # 2. Hybrid search Qdrant filtered by user_id
        q_vec = next(self.embedder.embed([query])).tolist()
        
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )
        
        hits = self.qdrant.search(
            collection_name="episodic_memory",
            query_vector=q_vec,
            query_filter=user_filter,
            limit=top_k
        )
        
        memories = [hit.payload["text"] for hit in hits]
        memories_str = "\n".join([f"- {m}" for m in memories]) if memories else "No relevant memories found."

        # 3. Assemble context string
        context = (
            f"--- ASSEMBLED CONTEXT FOR USER {user_id} ---\n"
            f"[USER PROFILE]: {profile_context}\n"
            f"[RECENT ACTIVITY]: {recent_context}\n"
            f"[EPISODIC MEMORIES]:\n{memories_str}\n"
            f"----------------------------------------"
        )
        
        return context
