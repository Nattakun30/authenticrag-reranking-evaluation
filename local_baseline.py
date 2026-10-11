from pathlib import Path
 
from llama_index.core import Document

from llama_index.core.node_parser import MarkdownNodeParser

from sentence_transformers import SentenceTransformer

from opensearchpy import OpenSearch, helpers
 
CORPUS_FILES = ["1.md", "2.md", "44.md", "5555.md"]
 
BM25_INDEX = "local-baseline-bm25"

VECTOR_INDEX = "local-baseline-vector"
 
 
def main():

    # 1. Connect to OpenSearch

    client = OpenSearch(

        hosts=[{"host": "localhost", "port": 9200}],

        use_ssl=False

    )
 
    if not client.ping():

        raise RuntimeError("OpenSearch is not running")
 
    # 2. Load documents

    documents = []
 
    for filename in CORPUS_FILES:

        path = Path("corpus_input") / filename
 
        content = path.read_text(encoding="utf-8")
 
        documents.append(

            Document(

                text=content,

                metadata={"source": filename}

            )

        )
 
    # 3. Split documents into chunks

    parser = MarkdownNodeParser(chunk_size=256)

    chunks = parser.get_nodes_from_documents(documents)
 
    print("Total Chunks:", len(chunks))
 
    # 4. Generate embeddings locally

    print("Loading BGE-M3...")

    model = SentenceTransformer("BAAI/bge-m3")
 
    texts = [chunk.get_content() for chunk in chunks]
 
    embeddings = model.encode(

        texts,

        batch_size=8,

        show_progress_bar=True

    )
 
    # 5. Prepare OpenSearch documents

    bm25_actions = []

    vector_actions = []
 
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        doc_id = f"doc_{i}"

        content = chunk.get_content()

        source = chunk.metadata.get("source", "")
 
        bm25_actions.append({

            "_op_type": "index",

            "_index": BM25_INDEX,

            "_id": doc_id,

            "_source": {

                "doc_id": doc_id,

                "content": content,

                "source": source

            }

        })
 
        vector_actions.append({

            "_op_type": "index",

            "_index": VECTOR_INDEX,

            "_id": doc_id,

            "_source": {

                "doc_id": doc_id,

                "content": content,

                "source": source,

                "embedding": embedding.tolist()

            }

        })
 
    # 6. Upload to OpenSearch

    print("Uploading BM25 documents...")

    bm25_success, _ = helpers.bulk(

        client,

        bm25_actions,

        raise_on_error=True

    )
 
    print("Uploading Vector documents...")

    vector_success, _ = helpers.bulk(

        client,

        vector_actions,

        raise_on_error=True

    )
 
    # 7. Refresh and verify

    client.indices.refresh(index=BM25_INDEX)

    client.indices.refresh(index=VECTOR_INDEX)
 
    bm25_count = client.count(index=BM25_INDEX)["count"]

    vector_count = client.count(index=VECTOR_INDEX)["count"]
 
    print("\n===== INDEXING RESULTS =====")

    print("BM25 Uploaded:", bm25_success)

    print("Vector Uploaded:", vector_success)

    print("BM25 Documents:", bm25_count)

    print("Vector Documents:", vector_count)
 
    if bm25_count == len(chunks) and vector_count == len(chunks):

        print("Local Baseline Indexing Success")

    else:

        print("Warning: Document count mismatch")
 
 
if __name__ == "__main__":

    main()
 