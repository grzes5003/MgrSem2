import json
import os
import time

import pandas as pd
from elasticsearch import Elasticsearch, helpers

from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import ElasticsearchRetriever, DensePassageRetriever


def load_data(path: str, elastic_passwd, es_path):
    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs=f"{es_path}\\config\\certs\\http_ca.crt",
        basic_auth=("elastic", elastic_passwd)
    )

    print("open json")
    with open(path, 'r', encoding='utf-8') as f:
        text = json.load(f)
        resp = helpers.bulk(
            client,
            text,
            index="legal",
            doc_type="_doc"
        )
        print(resp)
        ...


def load_elastic_doc(path: str, eds: ElasticsearchDocumentStore):
    print("open json")
    with open(path, 'r', encoding='utf-8') as f:
        text = list(map(lambda item: {
            "content": item["text"],
            "meta": {"article": item["_id"], "title": item["title"]}
        }, json.load(f)))
        eds.write_documents(text, index='legal', batch_size=1000)


def load_faiss_doc(path: str, store: FAISSDocumentStore, retr: DensePassageRetriever):
    print("open json")
    with open(path, 'r', encoding='utf-8') as f:
        text = json.load(f)
        text = list(map(lambda item: {
            "content": item["text"],
            "meta": {"article": item["_id"], "title": item["title"]}
        }, text))
        store.write_documents(text)
        print("written")
        store.update_embeddings(retr, batch_size=99)
        store.save('data/legal-questions/acts.faiss')
        retr.save("data/legal-questions/retriever.pt")


def load_questions(path: str, rng: range) -> [dict]:
    with open(path, "r", encoding='utf-8') as f:
        return list(filter(lambda item: int(item['_id']) in rng, json.load(f)))


def answer_questions(questions: [dict], retrievers: []):
    for question in questions:
        result = [[ans.content for ans in retriever.retrieve(query=question['text'], top_k=2)]
                  for name, retriever in retrievers]
        print(question)
        print(result)
        print({"==============": "================="})
        ...


def process_question(df: pd.DataFrame, question, retriever, n):
    start = time.perf_counter()
    possible = retriever.retrieve(query=question['text'], top_k=n)
    i = 0
    for answer in possible:
        year, position, art = answer.meta['article'].split('_')
        i += len(df[(df['question'] == question['text'])
                    & (df['year'].astype(str) == year)
                    & (df['position'].astype(str) == position)
                    & (df['art'].astype(str) == art)])
    end = time.perf_counter()
    return 1.0 * i / n, 1.0 * i / (i + n), end - start


def get_results(n, retrievers: [], df):
    results = []
    for question in questions:
        for name, retriever in retrievers:
            results.append((*process_question(df, question, retriever, n), name))
    return pd.DataFrame(data=results, columns=[f"Pr@{n}", f"Rc@{n}", "time", "type"])


if __name__ == '__main__':
    ELASTIC_PASSWORD = os.environ['ELASTIC_PASSWORD']
    if ELASTIC_PASSWORD is None:
        exit(11)

    ES_PATH = os.environ['ES_PATH']
    if ES_PATH is None:
        exit(11)

    elastic_store = ElasticsearchDocumentStore(host="localhost", username="elastic",
                                               password=ELASTIC_PASSWORD, index="legal")
    # faiss_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    #
    elastic_retriever = ElasticsearchRetriever(elastic_store)
    # faiss_retriever = DensePassageRetriever(
    #     document_store=faiss_store,
    #     query_embedding_model="enelpol/czywiesz-question",
    #     passage_embedding_model="enelpol/czywiesz-context",
    # )

    # load_elastic_doc('data/legal-questions/passages.json', elastic_store)
    # load_faiss_doc('data/legal-questions/passages.json', faiss_store, faiss_retriever)

    # load_data('data/legal-questions/passages.json', ELASTIC_PASSWORD, ES_PATH)

    questions = load_questions('data/legal-questions/questions.json', range(1114, 1134))

    # document_store = FAISSDocumentStore(sql_url="sqlite://./faiss_document_store.db")
    # retriever = DensePassageRetriever(document_store=document_store)

    faiss_store = FAISSDocumentStore.load(index_path="data/legal-questions/acts.faiss")

    faiss_retriever = DensePassageRetriever(
        document_store=faiss_store,
        query_embedding_model="enelpol/czywiesz-question",
        passage_embedding_model="enelpol/czywiesz-context",
    )

    retrievers = [["ELASTIC", elastic_retriever], ["FAISS", faiss_retriever]]

    # answer_questions(questions, retrievers)

    df = pd.read_csv('data/legal-questions/questions.csv')

    results = get_results(3, retrievers, df)
    results.to_csv("res_lab09_3.csv")
    results = get_results(1, retrievers, df)
    results.to_csv("res_lab09_1.csv")
    ...
