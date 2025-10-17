import os
import re
import math
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # fallback to simple tokenizer

try:
    import requests  # used for OpenAI REST API
except Exception:
    requests = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    title: Optional[str]
    text: str


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    index_in_doc: int
    text: str
    start_token: int
    end_token: int
    header_path: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievedContext:
    doc_id: str
    chunk_ids: List[str]
    context: str


def _get_tokenizer(encoding_name: str = "cl100k_base"):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        try:
            return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            return None


def tokenize(text: str) -> List[int]:
    enc = _get_tokenizer()
    if enc is None:
        # Fallback: crude whitespace split mapped to token ids by hashing
        tokens = []
        for word in re.findall(r"\S+", text):
            tokens.append(abs(hash(word)) % 10000)
        return tokens
    return enc.encode(text, allowed_special={"<|endoftext|>"})


def detokenize(tokens: List[int]) -> str:
    enc = _get_tokenizer()
    if enc is None:
        # We cannot reconstruct original text from fallback; return placeholder length
        return " ".join([f"T{t}" for t in tokens])
    return enc.decode(tokens)


def chunk_by_tokens(text: str, max_tokens: int = 1024, overlap_tokens: int = 0) -> List[Tuple[int, int]]:
    tokens = tokenize(text)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + max_tokens, n)
        spans.append((start, end))
        if end == n:
            break
        start = end - overlap_tokens if overlap_tokens > 0 else end
    return spans


_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_NUM_HEADER_RE = re.compile(r"^(?:\d+[\.)])+\s+(.*)$")


def parse_hierarchy(text: str, title: Optional[str]) -> List[Tuple[int, List[str]]]:
    """
    Parse hierarchy from text using simple markdown-like and numbered headers.
    Returns a list mapping line index to current header path.
    """
    lines = text.splitlines()
    header_path: List[str] = [title] if title else []
    path_by_line: List[Tuple[int, List[str]]] = []

    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line.strip())
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            # Ensure header_path has at most (level-1) existing entries
            if level - 1 < len(header_path):
                header_path = header_path[: level - 1]
            # Extend path to level-1 with empty placeholders if needed
            while len(header_path) < level - 1:
                header_path.append("")
            if level - 1 >= 0:
                header_path = header_path[: level - 1] + [content]
        else:
            m2 = _NUM_HEADER_RE.match(line.strip())
            if m2:
                content = m2.group(1).strip()
                # Treat numbered header as level 2
                if len(header_path) < 1 and title:
                    header_path = [title]
                if len(header_path) >= 1:
                    header_path = header_path[:1] + [content]
                else:
                    header_path = [content]

        path_by_line.append((i, list(filter(None, header_path))))

    return path_by_line


def assign_headers_to_chunks(text: str, chunk_spans: List[Tuple[int, int]], title: Optional[str]) -> List[List[str]]:
    # Map token index to line index approximately by characters proportion
    lines = text.splitlines()
    if not lines:
        return [[] for _ in chunk_spans]

    # Build cumulative char positions per line for coarse mapping
    cum_chars = [0]
    for ln in lines:
        cum_chars.append(cum_chars[-1] + len(ln) + 1)  # +1 for newline

    hierarchy_by_line = parse_hierarchy(text, title)
    # For each chunk, estimate which line range it covers using char proportions
    full_text_len = len(text)
    header_paths: List[List[str]] = []

    for (start_tok, end_tok) in chunk_spans:
        # Approximate char positions using token proportions
        start_char = int((start_tok / max(1, start_tok + (end_tok - start_tok))) * full_text_len)
        end_char = int((end_tok / max(1, start_tok + (end_tok - start_tok))) * full_text_len)

        # Find line indices
        def char_to_line(c: int) -> int:
            # binary search in cum_chars
            lo, hi = 0, len(cum_chars) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum_chars[mid] <= c < cum_chars[mid + 1]:
                    return mid
                if c >= cum_chars[mid + 1]:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        s_line = max(0, min(len(lines) - 1, char_to_line(start_char)))
        # Choose header path from the start line (closest previous header)
        header_path = []
        for i in range(s_line, -1, -1):
            header_path = hierarchy_by_line[i][1]
            if header_path:
                break
        header_paths.append(header_path)

    return header_paths


class EmbeddingClient:
    def embed(self, texts: List[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002", timeout: int = 60):
        if requests is None:
            raise RuntimeError("The 'requests' package is required for OpenAIEmbeddingClient. Please install it.")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # OpenAI embeddings support batching; keep batch modest to avoid timeouts
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        embeddings: List[List[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"model": self.model, "input": batch}
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI embedding error {resp.status_code}: {resp.text}")
            data = resp.json()
            # Ensure order
            batch_embeddings = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            embeddings.extend(batch_embeddings)
        return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HierarchicalRAG:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        chunk_size_tokens: int = 1024,
        chunk_overlap_tokens: int = 0,
        alpha_chunk: float = 0.7,
        beta_headers: float = 0.3,
    ) -> None:
        self.embedding_client = embedding_client
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.alpha_chunk = alpha_chunk
        self.beta_headers = beta_headers

        # In-memory index
        self.doc_store: Dict[str, Document] = {}
        self.chunks_by_doc: Dict[str, List[Chunk]] = {}
        self.header_embeddings_by_doc: Dict[str, Dict[str, List[float]]] = {}

    def add_documents(self, docs: List[Document]) -> None:
        # Store docs
        for d in docs:
            self.doc_store[d.doc_id] = d

        # Build chunks
        all_chunks: List[Chunk] = []
        for d in docs:
            spans = chunk_by_tokens(d.text, max_tokens=self.chunk_size_tokens, overlap_tokens=self.chunk_overlap_tokens)
            header_paths = assign_headers_to_chunks(d.text, spans, d.title)
            doc_chunks: List[Chunk] = []
            for idx, ((s, e), header_path) in enumerate(zip(spans, header_paths)):
                chunk_text_tokens = tokenize(d.text)[s:e]
                # Try to reconstruct text for embedding; if tokenizer exists we can detokenize else slice chars
                if _get_tokenizer() is not None:
                    chunk_text = detokenize(chunk_text_tokens)
                else:
                    # Fallback: proportional slice on characters
                    total_tokens = len(tokenize(d.text))
                    start_char = int((s / max(1, total_tokens)) * len(d.text))
                    end_char = int((e / max(1, total_tokens)) * len(d.text))
                    chunk_text = d.text[start_char:end_char]

                chunk = Chunk(
                    doc_id=d.doc_id,
                    chunk_id=f"{d.doc_id}::chunk_{idx}",
                    index_in_doc=idx,
                    text=chunk_text,
                    start_token=s,
                    end_token=e,
                    header_path=header_path,
                )
                doc_chunks.append(chunk)
                all_chunks.append(chunk)
            self.chunks_by_doc[d.doc_id] = doc_chunks

        # Compute embeddings for chunks in batches
        texts = [c.text for c in all_chunks]
        if texts:
            chunk_embeddings = self.embedding_client.embed(texts)
            for chunk, emb in zip(all_chunks, chunk_embeddings):
                chunk.embedding = emb

        # Compute header embeddings per doc (unique headers only)
        for d in docs:
            header_set = set()
            for c in self.chunks_by_doc[d.doc_id]:
                for h in c.header_path:
                    header_set.add(h)
            if d.title:
                header_set.add(d.title)
            header_list = sorted(h for h in header_set if h)
            embs = self.embedding_client.embed(header_list) if header_list else []
            self.header_embeddings_by_doc[d.doc_id] = {h: e for h, e in zip(header_list, embs)}

    def retrieve_for_document(self, query: str, doc_id: str, top_k: int = 5) -> RetrievedContext:
        if doc_id not in self.doc_store:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        chunks = self.chunks_by_doc.get(doc_id, [])
        if not chunks:
            return RetrievedContext(doc_id=doc_id, chunk_ids=[], context="")

        q_emb = self.embedding_client.embed([query])[0]
        header_embs = self.header_embeddings_by_doc.get(doc_id, {})

        scored: List[Tuple[float, Chunk]] = []
        for c in chunks:
            chunk_sim = cosine_similarity(q_emb, c.embedding or [])
            header_sim = 0.0
            if c.header_path and header_embs:
                header_sim = max((cosine_similarity(q_emb, header_embs.get(h, [])) for h in c.header_path if h in header_embs), default=0.0)
            score = self.alpha_chunk * chunk_sim + self.beta_headers * header_sim
            scored.append((score, c))

        # Select top-k by score, then order by original index
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:top_k]]
        top_chunks.sort(key=lambda c: c.index_in_doc)

        context = "\n\n".join(c.text for c in top_chunks)
        return RetrievedContext(doc_id=doc_id, chunk_ids=[c.chunk_id for c in top_chunks], context=context)

    def retrieve(self, query: str, top_k_per_doc: int = 5) -> Dict[str, RetrievedContext]:
        results: Dict[str, RetrievedContext] = {}
        for doc_id in self.doc_store.keys():
            results[doc_id] = self.retrieve_for_document(query, doc_id, top_k=top_k_per_doc)
        return results



def build_hierarchical_rag_with_openai(
    docs: List[Dict[str, str]],
    openai_api_key: Optional[str] = None,
    model: str = "text-embedding-ada-002",
    chunk_size_tokens: int = 1024,
    chunk_overlap_tokens: int = 0,
    alpha_chunk: float = 0.7,
    beta_headers: float = 0.3,
) -> HierarchicalRAG:
    client = OpenAIEmbeddingClient(api_key=openai_api_key, model=model)
    rag = HierarchicalRAG(
        embedding_client=client,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        alpha_chunk=alpha_chunk,
        beta_headers=beta_headers,
    )
    rag.add_documents([Document(doc_id=d["id"], title=d.get("title"), text=d["text"]) for d in docs])
    return rag


def example_usage():  #
    docs = [
        {
            "id": "doc1",
            "title": "Sample Document",
            "text": (
                "# Introduction\n"
                "This is an example document demonstrating hierarchical RAG.\n\n"
                "## Background\n"
                "Retrieval augmented generation benefits from structure.\n\n"
                "## Method\n"
                "We parse headers and chunk by tokens to build an index.\n\n"
                "## Experiments\n"
                "We evaluate with top-5 chunks per document.\n"
            ),
        }
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    rag = build_hierarchical_rag_with_openai(docs, openai_api_key=api_key)
    res = rag.retrieve_for_document("How do you build the index?", doc_id="doc1", top_k=5)
    print("Chunk IDs:", res.chunk_ids)
    print("Context:\n", res.context)


if __name__ == "__main__":  
    example_usage()


