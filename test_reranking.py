import time

from sentence_transformers import CrossEncoder

from test_retrieval import hybrid_search
 
QUESTION = "โรคกรดไหลย้อนมีอาการอย่างไร และมีสาเหตุจากอะไร?"
 
print("Loading Cross-Encoder...")

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
 
# Baseline: Top 5

start = time.perf_counter()

baseline = hybrid_search(QUESTION, k=5)

baseline_time = time.perf_counter() - start
 
# Improved: retrieve 20 candidates, then rerank

start = time.perf_counter()

candidates = hybrid_search(QUESTION, k=20)
 
pairs = [

    (QUESTION, source.get("content", ""))

    for doc_id, score, source in candidates

]
 
scores = reranker.predict(pairs)
 
reranked = sorted(

    zip(candidates, scores),

    key=lambda item: float(item[1]),

    reverse=True

)[:5]
 
improved_time = time.perf_counter() - start
 
print("\n===== BASELINE TOP 5 =====")

for rank, (doc_id, score, source) in enumerate(baseline, 1):

    print(f"{rank}. {doc_id} | {source.get('source')}")
 
print("\n===== RERANKED TOP 5 =====")

for rank, ((doc_id, rrf_score, source), rerank_score) in enumerate(reranked, 1):

    print(

        f"{rank}. {doc_id} | "

        f"Rerank Score: {float(rerank_score):.4f} | "

        f"{source.get('content', '')[:100]}"

    )
 
print("\n===== LATENCY =====")

print(f"Baseline: {baseline_time:.3f} seconds")

print(f"Improved: {improved_time:.3f} seconds")
 