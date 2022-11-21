import concurrent
import os

from elasticsearch import Elasticsearch, helpers
import multiprocessing as mp

from util import read_files_dir

if (ELASTIC_PASSWORD := os.environ['ELASTIC_PASSWORD']) is None:
    exit(11)

if (ES_PATH := os.environ['ES_PATH']) is None:
    exit(11)

client = Elasticsearch(
    "https://localhost:9200",
    ca_certs=f"{ES_PATH}\\config\\certs\\http_ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)


def setup_es():
    settings = {
        'analysis': {
            "analyzer": {
                "ustawy_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "morfologik_stem",
                        "synonyms_filter"
                    ]
                }
            },
            "filter": {
                "synonyms_filter": {
                    "type": "synonym",
                    "synonyms": [
                        "kpk => kodeks postępowania karnego",
                        "kpc => kodeks postępowania cywilnego",
                        "kk => kodeks karny",
                        "kc => kodeks cywilny"]
                }
            },
        }
    }

    mapping = {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "ustawy_analyzer"
            }
        }
    }

    if not client.indices.exists(index='ustawy'):
        resp0 = client.indices.create(index='ustawy', settings=settings, mappings=mapping)


def load_docs(path: str):
    files = list(read_files_dir(path))

    if client.count(index='ustawy')['count'] != len(files):
        print('creating documents from files')
        for file in files:
            doc = {'text': file[1]}
            if (res := client.index(index='ustawy', document=doc, id=file[0])['result']) == 'error':
                exit(13)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def send_data(_client: Elasticsearch, words: [str], name: str = 'sample'):
    items = map(lambda item: item.split(), words)
    items = filter(lambda item: len(item) == 4, items)
    items = list(map(lambda item: {'word': item[0]}, items))

    if (res := helpers.bulk(_client, items, stats_only=True)) == 'failed':
        print(f"error occurred while loading data to {name}: {res}")
        exit(13)


def load_file_to_doc(path: str, name: str = 'sample'):
    if client.indices.exists(index=name):
        return

    client.indices.create(index=name)
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    print(f"{len(lines)=}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_data, client, chunk, name) for chunk in chunks(lines, 1000)]
        for future in concurrent.futures.as_completed(futures):
            print(f"future {future} done")
