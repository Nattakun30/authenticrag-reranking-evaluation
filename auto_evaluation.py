import json
import math
import time
 
from sentence_transformers import CrossEncoder
from test_retrieval import hybrid_search
 
 
DATASET_FILE = "evaluation_dataset.json"
TOP_K = 5
CANDIDATE_K = 20
 
 
def recall_at_k(results, relevant, k=5):
    if not relevant:
        return 0.0
    return len(set(results[:k]) & relevant) / len(relevant)
 
 
def reciprocal_rank(results, relevant):
    for rank, doc_id in enumerate(results, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
 
 
def ndcg_at_k(results, relevant, k=5):
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(results[:k], start=1)
        if doc_id in relevant
    )
 
    ideal_count = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
 
    return dcg / idcg if idcg else 0.0
 
 
with open(DATASET_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)
 
print("Loading Cross-Encoder...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
 
records = []
 
for item in dataset:
    question_id = item["question_id"]
    question = item["question"]
    relevant = set(item["relevant_docs"])
 
    print(f"\nEvaluating {question_id}: {question}")
 
    # Retrieve once: same candidate pool for fair comparison
    start = time.perf_counter()
    candidates = hybrid_search(question, k=CANDIDATE_K)
    retrieval_time = time.perf_counter() - start
 
    baseline_ids = [
        doc_id for doc_id, score, source in candidates[:TOP_K]
    ]
 
    pairs = [
        (question, source.get("content", ""))
        for doc_id, score, source in candidates
    ]
 
    start = time.perf_counter()
    scores = reranker.predict(pairs)
    rerank_time = time.perf_counter() - start
 
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )
 
    improved_ids = [
        candidate[0]
        for candidate, score in ranked[:TOP_K]
    ]
 
    for method, results, latency in [
        ("Baseline", baseline_ids, retrieval_time),
        ("Improved", improved_ids, retrieval_time + rerank_time),
    ]:
        records.append({
            "question_id": question_id,
            "method": method,
            "recall_at_5": recall_at_k(results, relevant),
            "mrr": reciprocal_rank(results, relevant),
            "ndcg_at_5": ndcg_at_k(results, relevant),
            "latency_sec": latency,
            "retrieved_docs": results,
        })
 
    print("Baseline:", baseline_ids)
    print("Improved:", improved_ids)
 
 
print("\n========== AVERAGE RESULTS ==========")
 
for method in ["Baseline", "Improved"]:
    rows = [r for r in records if r["method"] == method]
 
    print(f"\n{method}")
    for metric in [
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
        "latency_sec",
    ]:
        average = sum(r[metric] for r in rows) / len(rows)
        print(f"{metric}: {average:.4f}")
 
 
with open("evaluation_results.json", "w", encoding="utf-8") as file:
    json.dump(records, file, ensure_ascii=False, indent=2)
 
print("\nSaved: evaluation_results.json")