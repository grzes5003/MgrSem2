from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.naive_bayes import GaussianNB
import fasttext

from transformers import DataCollatorForTokenClassification, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
from transformers import pipeline


def bayesian_classifier(dataset: Dataset):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataset['train']['text'])
    tgidf_train, tgidf_test = vectorizer.transform(dataset['train']['text']).toarray(), \
        vectorizer.transform(dataset['test']['text']).toarray()

    gnb1 = GaussianNB()
    gnb1.fit(tgidf_train, dataset['train']['label'])

    # evaluate
    return evaluate_bayesian(gnb1.predict(tgidf_test), dataset['test']['label'], 'GaussianNaiveBayes')


def evaluate_bayesian(predicted, actual, name: str, ver: int = 1):
    if ver == 1:
        return {"name": name, "accuracy": accuracy_score(predicted, actual),
                "precision": precision_score(predicted, actual),
                "recall": recall_score(predicted, actual), "f1": f1_score(predicted, actual)}
    return {"name": name, "accuracy": accuracy_score(predicted, actual),
            "precision": precision_score(predicted, actual, average='macro'),
            "recall": recall_score(predicted, actual, average='macro'),
            "f1": f1_score(predicted, actual, average='macro')}


def fasttext_format(name: str, dataset: Dataset):
    items = ['train', 'test']
    for item in items:
        with open(f'{name}_{item}.txt', "w", encoding="utf-8") as f:
            for label, text in zip(dataset[item]['label'], dataset[item]['text']):
                f.write(f"__label__{label} {text}\n")


def fasttext_classifier(filename: str, path: str = '.'):
    model = fasttext.train_supervised(f'{path}/{filename}_train.txt')
    result = model.test(f'{path}/{filename}_test.txt')
    return {"name": "FastText", "accuracy": result[1], "precision": None, "recall": None, "f1": None}


def transformer_classifier(dataset: Dataset, name: str, num_labels: int):
    _pipeline = pipeline('fill-mask', model='allegro/herbert-base-cased')
    t_train, t_test = _pipeline.tokenizer(dataset['train']['text']), \
        _pipeline.tokenizer(dataset['test']['text'])

    data_collator = DataCollatorForTokenClassification(_pipeline.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained('allegro/herbert-base-cased', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='results/lab07',
        learning_rate=0.00002,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=model.tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_pretrained(name)
    return model


if __name__ == '__main__':
    dataset1 = load_dataset("poleval2019_cyberbullying", "task01")
    dataset2 = load_dataset("poleval2019_cyberbullying", "task02")

    results = []

    # bayesian_classifier(dataset1)
    # fasttext_format('ds1', dataset1)
    # fasttext_format('ds2', dataset2)

    # results.append(fasttext_classifier('ds1', 'data_lab07'))
    # results.append(fasttext_classifier('ds2', 'data_lab07'))

    transformer_classifier(dataset1, "herbert-base-cased-bullying-1", 2)
    transformer_classifier(dataset2, "herbert-base-cased-bullying-2", 3)

    ...
