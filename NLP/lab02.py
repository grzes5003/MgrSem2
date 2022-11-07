from typing import Mapping, Any

from elastic_transport import ObjectApiResponse
from util import read_files_dir, client, setup_es, load_docs


def search(query: Mapping[str, Any], *, highlight: Mapping[str, Any] = None) -> ObjectApiResponse:
    return client.search(
        index="ustawy",
        body={"query": query, "highlight": highlight}
        if highlight else {"query": query}
    )


def exc06() -> int:
    """
    Number of legislative acts containing the word ustawa (in any form)
    :return:
    """
    query = {
        "match": {
            "text": {
                "query": "ustawa"
            }
        }}
    return search(query)["hits"]["total"]


def exc07() -> Any:
    """
    Number of occurrences of the word ustawa by searching for this particular form,
    including the other inflectional forms
    :return:
    """
    return client.termvectors(index="ustawy",
                             id="1997_629.txt",
                             body={
                                 "fields": ["text"],
                                 "term_statistics": True,
                                 "field_statistics": True
                             })["term_vectors"]["text"]["terms"]["ustawa"]


def exc09() -> int:
    """
    Number of legislative acts containing the words
    kodeks postępowania cywilnego in the specified order, but in any inflection form.
    :return:
    """
    query = {
        "match_phrase": {
            "text": {
                "query": "kodeks postępowania cywilnego"
            }
        }}
    return search(query)["hits"]["total"]


def exc10() -> int:
    """
    Number of legislative acts containing the words
    wchodzi w życie (in any form) allowing for up to 2 additional words in the searched phrase.
    :return:
    """
    query = {
        "match_phrase": {
            "text": {
                "query": "wchodzi w życie",
                "slop": 2
            }
        }}
    return search(query)["hits"]["total"]


def exc11() -> [str]:
    """
    10 documents that are the most relevant for the phrase konstytucja
    :return:
    """
    query = {
        "match": {
            "text": {
                "query": "konstytucja",
            }
        }}
    highlight = {
        "fields": {
            "text": {}
        },
        "boundary_scanner": "sentence",
        "order": "score"
    }
    res = search(query, highlight=highlight)["hits"]
    return [result['_id'] for result in res['hits']][:10]


def exc12():
    """
    Print the excerpts containing the word konstytucja (up to three excerpts per document) from the previous task.
    :return:
    """
    query = {
        "match": {
            "text": {
                "query": "konstytucja",
            }
        }}
    highlight = {
        "fields": {
            "text": {}
        },
        "boundary_scanner": "sentence",
        "order": "score",
        "number_of_fragments": 3
    }
    res = search(query, highlight=highlight)["hits"]
    for _id, texts in [(result['_id'], result['highlight']['text']) for result in res['hits']][:10]:
        print(f'-------------- {_id}')
        print(texts)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    setup_es()
    load_docs(path)

    print(f'{exc06()=}')
    print(f'{exc07()=}')
    print(f'{exc09()=}')
    print(f'{exc10()=}')
    print(f'{exc11()=}')
    print(f'{exc12()=}')

    print(client.info())
