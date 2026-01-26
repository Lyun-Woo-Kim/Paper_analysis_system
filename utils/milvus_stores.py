from __future__ import annotations

import os
from typing import List
from langchain_core.documents import Document

from langchain_milvus import Milvus as milvus_store
from langchain_huggingface import HuggingFaceEmbeddings

class PaperMilvusStoresLite:
    """
    - 로컬 Milvus Lite 파일 1개 = 논문 1개 (papers/{paper_name}.db)
    - 컬렉션 3개 = text/equation/visual
    """
    def __init__(self, paper_name: str, embedding_model_name: str, base_dir: str = "milvusDBs", drop_old: bool = False):
        self.paper_name_raw = paper_name
        self.paper_name = paper_name.replace(" ", "_")

        os.makedirs(base_dir, exist_ok=True)
        self.db_path = os.path.join(base_dir, f"{self.paper_name}.db")

        # Milvus Lite: 로컬 파일로 저장 (uri에 DB 파일 경로)
        self.connection_args = {"uri": self.db_path}
        self.embedding = HuggingFaceEmbeddings(
            model_name = embedding_model_name,
            encode_kwargs = {'normalize_embeddings': True})

        self.text_collection = f"{self.paper_name}_text"
        self.eq_collection = f"{self.paper_name}_equation"
        self.visual_collection = f"{self.paper_name}_visual"

        self.text_store = milvus_store(
            collection_name=self.text_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=drop_old,
        )
        self.eq_store = milvus_store(
            collection_name=self.eq_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=drop_old,
        )
        self.visual_store = milvus_store(
            collection_name=self.visual_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=drop_old,
        )

    def add_text_chunks(self, chunks: List[dict], doc_id: str):
        docs = []
        for c in chunks:
            # bbox를 사용하고 싶으면 아래 주석 사용
            # meta = {"doc_id": doc_id, "page_index": int(c["page_index"]), "type": "text"}
            
            meta = {"doc_id": doc_id, "type": "text"}
            docs.append(Document(page_content=c["text"], metadata=meta))
        if docs:
            self.text_store.add_documents(docs)

    def add_equations(self, eq_items: List[dict], doc_id: str):
        docs = []
        for e in eq_items:
            meta = {
                "doc_id": str(doc_id),
                "page_index": int(e.get("page_index", 0)),
                "bbox": str(e.get("bbox", [])),
                "type": "equation",
                "tag": e.get("tag") or "", 
                "tag_raw": e.get("tag_raw") or "",
                "confidence": float(e.get("confidence", 0.0)),
                "symbol": str(e.get("symbol", {}))
            }
            content = (
                f"LATEX: {e['latex']}\n"
                f"TAG: {e.get('tag')}\n"
                f"EXPLANATION: {e.get('analysis','')}\n"
            )
            docs.append(Document(page_content=content, metadata=meta))
        if docs:
            self.eq_store.add_documents(docs)

    def add_visuals(self, visual_items: List[dict], doc_id: str):
        docs = []
        for v in visual_items:
            meta = {
                "doc_id": doc_id,
                "page_index": int(v["page_index"]),
                "bbox": str(v.get("bbox", [])),
                "label": v.get("label", ""),
                "type": "visual",
            }
            content = (
                f"CAPTION: {v.get('caption','')}\n"
                f"SUMMARY: {v.get('summary','')}\n"
                f"KEY_POINTS: {v.get('key_points','')}"
            )
            docs.append(Document(page_content=content, metadata=meta))
        if docs:
            self.visual_store.add_documents(docs)
            
    def reset_database(self):
        """
        DB 초기화 함수
        현재 연결된 DB 파일의 모든 컬렉션(text, equation, visual)을 삭제하고 새로 만듭니다.
        저장된 데이터가 모두 사라집니다.
        """
        print(f"⚠ WARNING: Resetting entire database for '{self.paper_name}'...")
        
        # 1. Text 컬렉션 초기화
        self.text_store = milvus_store(
            collection_name=self.text_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=True  # <--- 핵심: 기존 거 날리고 새로 만듦
        )
        
        # 2. Equation 컬렉션 초기화
        self.eq_store = milvus_store(
            collection_name=self.eq_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=True
        )
        
        # 3. Visual 컬렉션 초기화
        self.visual_store = milvus_store(
            collection_name=self.visual_collection,
            connection_args=self.connection_args,
            embedding_function=self.embedding,
            drop_old=True
        )
        
        print(f"✅ Reset Complete. All collections are empty.")