import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add fallback path if keys are missing or API calls fail
class CatalogManager:
    def __init__(self):
        self.catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", 
            "shl_product_catalog.json"
        )
        self.embeddings_cache_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", 
            "shl_catalog_embeddings.json"
        )
        self.products: List[Dict[str, Any]] = []
        self.products_by_id: Dict[str, Dict[str, Any]] = {}
        self.load_catalog()
        
        # Determine embedding provider based on environment
        self.provider = "tfidf"
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        if self.openai_key:
            self.provider = "openai"
        elif self.gemini_key:
            self.provider = "gemini"
            
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embeddings_matrix = None
        
        # Fit local TF-IDF vectorizer as a baseline/fallback
        self.init_tfidf()
        
        # Attempt to load or generate vector embeddings
        self.init_vector_embeddings()

    def load_catalog(self):
        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                # Use strict=False to bypass raw control characters in catalog
                self.products = json.loads(f.read(), strict=False)
                # Clean up durations and languages
                for p in self.products:
                    # Normalise duration
                    if not p.get("duration") and p.get("duration_raw"):
                        # Extract "minutes = 30" or similar
                        p["duration"] = p["duration_raw"].replace("Approximate Completion Time in minutes = ", "").strip() + " mins"
                    elif not p.get("duration"):
                        p["duration"] = "Varies"
                    
                    if not p.get("languages") and p.get("languages_raw"):
                        p["languages"] = [lang.strip() for lang in p["languages_raw"].split(",") if lang.strip()]
                    
                    self.products_by_id[p["entity_id"]] = p
            print(f"CatalogManager: Loaded {len(self.products)} products from catalog.")
        except Exception as e:
            print(f"CatalogManager Error loading catalog: {e}", file=sys.stderr)
            self.products = []

    def init_tfidf(self):
        if not self.products:
            return
        
        texts = []
        for p in self.products:
            # Combine attributes to build a search text blob
            name = p.get("name", "")
            desc = p.get("description", "")
            keys = " ".join(p.get("keys", []))
            job_levels = " ".join(p.get("job_levels", []))
            languages = " ".join(p.get("languages", []))
            
            # Boost the name to make keyword matches on name strong
            text = f"{name} {name} {name} {desc} {keys} {job_levels} {languages}"
            texts.append(text)
            
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print("CatalogManager: Local TF-IDF search initialized.")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        if self.provider == "openai" and self.openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key)
                response = client.embeddings.create(
                    input=[text],
                    model="text-embedding-3-small"
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"CatalogManager: OpenAI Embedding error: {e}", file=sys.stderr)
                
        elif self.provider == "gemini" and self.gemini_key:
            try:
                # Use google-genai or standard HTTP request
                # To be simple and robust, let's use the google-genai API or standard REST
                import urllib.request
                import json
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.gemini_key}"
                data = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text}]}
                }
                req = urllib.request.Request(
                    url, 
                    data=json.dumps(data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    return res['embedding']['values']
            except Exception as e:
                print(f"CatalogManager: Gemini Embedding error: {e}", file=sys.stderr)
                
        return None

    def init_vector_embeddings(self):
        if self.provider == "tfidf" or not self.products:
            return
            
        # Try to load from cache
        cache_loaded = False
        if os.path.exists(self.embeddings_cache_path):
            try:
                with open(self.embeddings_cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    if cache.get("provider") == self.provider and len(cache.get("embeddings", {})) == len(self.products):
                        # Convert cache back to numpy matrix
                        embeddings_list = []
                        for p in self.products:
                            eid = p["entity_id"]
                            if eid in cache["embeddings"]:
                                embeddings_list.append(cache["embeddings"][eid])
                            else:
                                break
                        
                        if len(embeddings_list) == len(self.products):
                            self.embeddings_matrix = np.array(embeddings_list)
                            cache_loaded = True
                            print(f"CatalogManager: Loaded {len(self.products)} cached embeddings using {self.provider}.")
            except Exception as e:
                print(f"CatalogManager: Error loading embeddings cache: {e}", file=sys.stderr)
                
        if not cache_loaded:
            print(f"CatalogManager: Generating new embeddings using {self.provider} (this might take a minute)...")
            embeddings_dict = {}
            embeddings_list = []
            
            # We batch or loop
            success_count = 0
            for i, p in enumerate(self.products):
                # Build representation text
                name = p.get("name", "")
                desc = p.get("description", "")
                keys = ", ".join(p.get("keys", []))
                text = f"Name: {name}. Description: {desc}. Types: {keys}."
                
                emb = self.get_embedding(text)
                if emb is not None:
                    embeddings_dict[p["entity_id"]] = emb
                    embeddings_list.append(emb)
                    success_count += 1
                else:
                    # Fallback to tfidf if embeddings generation fails
                    print(f"CatalogManager: Embedding failed for product {p['entity_id']}. Falling back to TF-IDF.")
                    break
                    
            if success_count == len(self.products):
                self.embeddings_matrix = np.array(embeddings_list)
                # Save cache
                try:
                    with open(self.embeddings_cache_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "provider": self.provider,
                            "embeddings": embeddings_dict
                        }, f)
                    print(f"CatalogManager: Saved {len(self.products)} embeddings to cache.")
                except Exception as e:
                    print(f"CatalogManager: Error writing embeddings cache: {e}", file=sys.stderr)
            else:
                print("CatalogManager: Embedded generation incomplete. Falling back to local TF-IDF vector search.")
                self.provider = "tfidf"

    def search(self, query: str, top_k: int = 10, 
               job_levels: Optional[List[str]] = None, 
               keys: Optional[List[str]] = None, 
               languages: Optional[List[str]] = None,
               test_types: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes a hybrid semantic / TF-IDF search. Returns matching items and cosine similarity scores.
        """
        if not self.products:
            return []
            
        # 1. Compute similarity scores
        if self.provider != "tfidf" and self.embeddings_matrix is not None:
            query_emb = self.get_embedding(query)
            if query_emb is not None:
                query_vec = np.array(query_emb).reshape(1, -1)
                scores = cosine_similarity(query_vec, self.embeddings_matrix)[0]
            else:
                # API failure fallback
                query_tfidf = self.tfidf_vectorizer.transform([query])
                scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        else:
            query_tfidf = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
        # 2. Match and filter products
        results = []
        for idx, score in enumerate(scores):
            product = self.products[idx]
            match_score = float(score)
            
            # Simple keyword boosting: if exact name words match, boost the score
            query_words = set(query.lower().split())
            name_words = set(product.get("name", "").lower().split())
            common_words = query_words.intersection(name_words)
            if common_words:
                match_score += 0.15 * len(common_words)
            
            # Normalize match score between 0 and 100 for user friendly display
            # Typically cosine similarities are in range [0, 0.8], so we scale appropriately
            scaled_score = min(int(match_score * 100), 100)
            scaled_score = max(scaled_score, 10) # minimum 10% match
            
            # Filters
            if job_levels:
                product_levels = [l.lower() for l in product.get("job_levels", [])]
                if not any(jl.lower() in product_levels for jl in job_levels):
                    continue
                    
            if keys:
                product_keys = [k.lower() for k in product.get("keys", [])]
                if not any(key.lower() in product_keys for key in keys):
                    continue
                    
            if test_types:
                product_keys = [k.lower() for k in product.get("keys", [])]
                if not any(tt.lower() in product_keys for tt in test_types):
                    continue
                    
            if languages:
                product_langs = [l.lower() for l in product.get("languages", [])]
                if not any(lang.lower() in product_langs for lang in languages):
                    continue
                    
            results.append((product, scaled_score))
            
        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def manual_filter(self, query: Optional[str] = None, 
                      duration: Optional[str] = None,
                      language: Optional[str] = None,
                      test_type: Optional[str] = None,
                      skill: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Supports the manual Assessment Library filtering
        """
        filtered = self.products
        
        if query:
            q = query.lower()
            filtered = [
                p for p in filtered 
                if q in p.get("name", "").lower() or q in p.get("description", "").lower()
            ]
            
        if duration:
            # e.g., "under 15 mins", "15-30 mins", "over 30 mins"
            d_val = duration.lower()
            temp = []
            for p in filtered:
                dur_str = p.get("duration", "").lower()
                # Parse duration numbers
                digits = "".join(c for c in dur_str if c.isdigit())
                if digits:
                    mins = int(digits)
                    if "under 15" in d_val and mins < 15:
                        temp.append(p)
                    elif "15-30" in d_val and 15 <= mins <= 30:
                        temp.append(p)
                    elif "over 30" in d_val and mins > 30:
                        temp.append(p)
                    elif "varies" in d_val:
                        temp.append(p)
                else:
                    if "varies" in d_val:
                        temp.append(p)
            filtered = temp
            
        if language:
            l_val = language.lower()
            filtered = [
                p for p in filtered 
                if any(l_val in lang.lower() for lang in p.get("languages", []))
            ]
            
        if test_type:
            tt_val = test_type.lower()
            filtered = [
                p for p in filtered 
                if any(tt_val in k.lower() for k in p.get("keys", []))
            ]
            
        if skill:
            s_val = skill.lower()
            filtered = [
                p for p in filtered 
                if s_val in p.get("description", "").lower() or any(s_val in k.lower() for k in p.get("keys", []))
            ]
            
        return filtered

# Create global instance
catalog_manager = CatalogManager()
