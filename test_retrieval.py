from opensearchpy import OpenSearch

from sentence_transformers import SentenceTransformer
 
BM25_INDEX = "local-baseline-bm25"

VECTOR_INDEX = "local-baseline-vector"
 
client = OpenSearch(

    hosts=[{"host": "localhost", "port": 9200}],

    use_ssl=False

)
 
model = SentenceTransformer("BAAI/bge-m3")
 
 
def sparse_search(query, k=10):

    response = client.search(

        index=BM25_INDEX,

        body={

            "query": {

                "match": {

                    "content": query

                }

            },

            "size": k

        }

    )

    return [

        (hit["_id"], hit["_score"], hit["_source"])

        for hit in response["hits"]["hits"]

    ]
 
 
def dense_search(query, k=10):

    embedding = model.encode(query).tolist()
 
    response = client.search(

        index=VECTOR_INDEX,

        body={

            "query": {

                "knn": {

                    "embedding": {

                        "vector": embedding,

                        "k": k

                    }

                }

            },

            "size": k

        }

    )

    return [

        (hit["_id"], hit["_score"], hit["_source"])

        for hit in response["hits"]["hits"]

    ]
 
 
def rrf_fusion(rankings, rrf_k=60):

    scores = {}

    sources = {}
 
    for results in rankings:

        for rank, (doc_id, _, source) in enumerate(results):

            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)

            sources[doc_id] = source
 
    return [

        (doc_id, score, sources[doc_id])

        for doc_id, score in sorted(

            scores.items(),

            key=lambda item: item[1],

            reverse=True

        )

    ]
 
 
def hybrid_search(query, k=5):

    sparse = sparse_search(query, k=10)

    dense = dense_search(query, k=10)
 
    fused = rrf_fusion([sparse, dense])
 
    return fused[:k]
 
 
if __name__ == "__main__":

    question = "โรคกรดไหลย้อนมีอาการอย่างไร และมีสาเหตุจากอะไร?"
 
    results = hybrid_search(question)
 
    print("\nQUESTION:", question)

    print("\n===== TOP 5 RESULTS =====")
 
    for rank, (doc_id, score, source) in enumerate(results, start=1):

        print(f"\nRank: {rank}")

        print(f"Document ID: {doc_id}")

        print(f"RRF Score: {score:.6f}")

        print(f"Source File: {source.get('source')}")

        print(f"Content: {source.get('content', '')[:250]}")
 