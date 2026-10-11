import json

from opensearchpy import OpenSearch, helpers
 
client = OpenSearch(

    hosts=[{"host": "localhost", "port": 9200}],

    use_ssl=False

)
 
chunks = []
 
for hit in helpers.scan(

    client,

    index="local-baseline-bm25",

    query={"query": {"match_all": {}}}

):

    source = hit["_source"]
 
    chunks.append({

        "doc_id": hit["_id"],

        "source": source.get("source", ""),

        "content": source.get("content", "")

    })
 
chunks.sort(

    key=lambda item: int(item["doc_id"].split("_")[-1])

)
 
with open("chunks_export.json", "w", encoding="utf-8") as f:

    json.dump(chunks, f, ensure_ascii=False, indent=2)
 
print("Total Exported Chunks:", len(chunks))

print("Export completed: chunks_export.json")
 
for chunk in chunks:

    preview = chunk["content"].replace("\n", " ")[:100]

    print(

        f"{chunk['doc_id']} | "

        f"{chunk['source']} | "

        f"{preview}"

    )
 