from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
from haystack.nodes import ElasticsearchRetriever, DensePassageRetriever

if __name__ == '__main__':
    elastic_store = ElasticsearchDocumentStore()
    faiss_store = FAISSDocumentStore(faiss_index_factory_str="Flat")

    elastic_retriever = ElasticsearchRetriever(elastic_store)
    faiss_retriever = DensePassageRetriever(
        document_store=faiss_store,
        query_embedding_model="enelpol/czywiesz-question",
        passage_embedding_model="enelpol/czywiesz-context",
    )
