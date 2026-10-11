import math
 
# Ground Truth ชั่วคราวจากคำถามที่ตรวจสอบแล้ว

RELEVANT_DOCS = {"doc_45", "doc_43"}
 
BASELINE = [

    "doc_42",

    "doc_45",

    "doc_46",

    "doc_43",

    "doc_48"

]
 
IMPROVED = [

    "doc_45",

    "doc_43",

    "doc_42",

    "doc_48",

    "doc_47"

]
 
 
def recall_at_k(results, relevant, k=5):

    retrieved = set(results[:k])

    return len(retrieved & relevant) / len(relevant) if relevant else 0.0
 
 
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
 
 
def evaluate(name, results):

    print(f"\n===== {name} =====")

    print(f"Recall@5: {recall_at_k(results, RELEVANT_DOCS):.4f}")

    print(f"MRR:      {reciprocal_rank(results, RELEVANT_DOCS):.4f}")

    print(f"nDCG@5:   {ndcg_at_k(results, RELEVANT_DOCS):.4f}")
 
 
evaluate("BASELINE", BASELINE)

evaluate("IMPROVED", IMPROVED)
 