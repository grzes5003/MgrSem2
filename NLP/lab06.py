import itertools
from typing import Tuple, Any, List

from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch


def fill_mask(models, sentences):
    for unmasker in models:
        for sentence in sentences:
            MASK_TOKEN = unmasker.tokenizer.mask_token
            result = unmasker(sentence.replace('<mask>', MASK_TOKEN), top_k=5)
            print("\n---------------")
            print(f"{unmasker.tokenizer.name_or_path}\n")
            if isinstance(result[0], list):
                result = list(itertools.chain(*result))
            for res in result:
                print(f"{res['score']}: {res['sequence']}")


def exc02():
    """
    Download three Polish models from the Huggingface repository.
    """
    return [pipeline('fill-mask', model=name)
            for name in [
                "Twitter/twhin-bert-base",
                # "facebook/xlm-roberta-xl",
                "dkleczek/bert-base-polish-uncased-v1",
                "allegro/herbert-base-cased"]]


def exc03(models: List[Tuple[Any, Any]]):
    """
    Devise a method to test if the langage model understands Polish cases
    """
    sentences = [
        "Warszawa to największe <mask>",
        "Warszawa to największa <mask>",
        "Nie ma już <mask> w sklepie",
        "Klient przygląda się <mask>",
        "Na podłodze widać <mask>",
        "Programista miał problemy z <mask>",
        "Podczas konferencji prasowej była mowa o <mask>",
        "Hej ty, <mask>"
    ]

    fill_mask(models, sentences)


def exc04(models):
    """
    Devise a method to test long-range relationships such as gender
    """
    sentences = [
        "On ją okłamał, przez co ona <mask>",
        "Ona go okłamała, przez co on <mask>",

        "On w wolnym czasie lubił <mask>",
        "Ona w wolnym czasie lubiła <mask>",

        "Nie było już dostępnych <mask> dla niej",
        "Nie było już dostępnych <mask> dla niego",
    ]

    fill_mask(models, sentences)


def exc05(models):
    """
    Check if the model captures real-world knowledge
    """
    sentences = [
        "<mask> wrze w temperaturze 100 stopni, a zamarza w temperaturze 0 stopni Celsjusza.",
        "<mask> jest miejscem gdzie się leczy ludzi",
        "programowanie <mask> jest najlepszym paradygmetem programowania"
    ]

    fill_mask(models, sentences)


if __name__ == '__main__':
    models = exc02()
    print('loaded models')

    # exc03(models)
    # exc04(models)
    exc05(models)
