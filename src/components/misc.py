import json

def json_print(data):
    print(json.dumps(data, indent=2))

def dense_search(client, query, top_k=2):
    response = (
        client.query
        .get("Chatbot", ["content", "metadata"])
        .with_near_text({"concepts":[query]})
        .with_limit(top_k)
        .do()
    )
    return response


def sparse_search(client, query, top_k=3):
    response = (
        client.query
        .get("Chatbot",["content","metadata"])
        .with_bm25(query=query)
        .with_limit(top_k)
        .do()
    )
    return response


def hybrid_search(client, query, top_k=3):
    response = (
        client.query
        .get("Chatbot",["content","metadata"])
        .with_hybrid(query=query, alpha=0.5)
        .with_limit(top_k)
        .do()
    )
    return response