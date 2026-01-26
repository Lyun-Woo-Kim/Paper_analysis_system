from __future__ import annotations
from typing import Sequence, Optional, Dict, Any, Literal, List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

from pathlib import Path
import sys
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.insert(0, str(root_dir))

from utils.milvus_stores import PaperMilvusStoresLite

Collection = Literal["text", "equation", "visual"]

class PaperInfo_RAG: 
    def __init__(self, configs: str, re_rank: bool = False): 
        self.config = configs
        self.milvus_store = PaperMilvusStoresLite(self.config["PAPER_NAME"], self.config["EMBEDDING_MODEL_NAME"])
        self.llm = self.load_langchain_model()
        self.doc_id = self.config["PAPER_NAME"]
        self.expr = None
        self.re_rank = re_rank
        if re_rank: 
            from sentence_transformers import CrossEncoder
            self.re_rank_model = CrossEncoder("BAAI/bge-reranker-base")
        
    
    def load_langchain_model(self) -> ChatOpenAI: 
        model = ChatOpenAI(
            model=self.config['LLM_MODEL_NAME'],\
            openai_api_base=self.config['API_BASE'],
            api_key="EMPTY", 
            max_tokens=self.config['LLM_MAX_NEW_TOKENS'],
            temperature=self.config['LLM_TEMPERATURE'],
            top_p=self.config['LLM_TOP_P']
        )
        return model

    def load_fix_langchain_model(self) -> ChatOpenAI: 
        model = ChatOpenAI(
            model=self.config['LLM_MODEL_NAME'],
            openai_api_base=self.config['API_BASE'],
            api_key="EMPTY", 
            max_tokens=self.config['LLM_MAX_NEW_TOKENS'],
            temperature = 0,
            seed=42
        )
        return model
    
    def re_rank_docs(self, final_num_docs: int, question: str, docs: List[Document]) -> List[Document]:
        if self.re_rank: 
            pairs = [[question, d.page_content] for d in docs]
            scores = self.re_rank_model.predict(pairs)
            docs_with_scores = list(zip(docs, scores))
            docs_with_scores.sort(key=lambda x:x[1], reverse = True)
            return [doc for doc, _ in docs_with_scores[:final_num_docs]]
    
    def build_rag_chain(self, 
        prompt_template: str,
        retriever_k_dict: dict, 
        collections: Sequence[Collection] = ["text", "equation", "visual"],
        ):
        """
        LangChain Runnable (chain)
        """
        # Re-Rank 사용 시, 쿼리 수 3배 증가.
        if self.re_rank: 
            retriever_k_dict = {col: retriever_k_dict[col] * 3 for col in collections}
            
        def retrieve_ctx(inputs: Dict[str, Any]) -> Dict[str, Any]:
            q = inputs["question"]
            parts = []
            if self.expr is None and self.doc_id is not None:
                self.expr = f'doc_id == "{self.doc_id}"'

            for col in collections:
                vs = {
                    "text": self.milvus_store.text_store,
                    "equation": self.milvus_store.eq_store,
                    "visual": self.milvus_store.visual_store,
                }[col]

                if self.expr:
                    docs = vs.similarity_search(q, k=retriever_k_dict[col], expr=self.expr)
                else:
                    docs = vs.similarity_search(q, k=retriever_k_dict[col])
                    
                if not docs:
                    parts.append(f"[{col.upper()}] (no results)")
                    continue
                
                if self.re_rank: 
                    docs = self.re_rank_docs(int(retriever_k_dict[col]/3), q, docs)

                chunk_lines = []
                for d in docs:
                    md = d.metadata or {}
                    chunk_lines.append(
                        f"[{col.upper()}] page={md.get('page_index')} bbox={md.get('bbox')} "
                        f"label={md.get('label')} tag={md.get('tag')}\n{d.page_content}"
                    )
                parts.append("\n\n".join(chunk_lines))

            context = "\n\n".join(parts).strip()
            return {"context": context, "question": q}

        retriever = RunnableLambda(retrieve_ctx)

        prompt = PromptTemplate.from_template(prompt_template)
        chain = retriever | prompt | self.llm | StrOutputParser()
        return chain