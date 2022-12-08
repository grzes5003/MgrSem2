import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
import fasttext
import evaluate
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import confusion_matrix

from transformers import DataCollatorForTokenClassification, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer, AutoTokenizer, DataCollatorWithPadding
from transformers import pipeline


def bayesian_classifier(dataset: Dataset, ver):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataset['train']['text'])
    tgidf_train, tgidf_test = vectorizer.transform(dataset['train']['text']).toarray(), \
        vectorizer.transform(dataset['test']['text']).toarray()

    gnb1 = GaussianNB()
    gnb1.fit(tgidf_train, dataset['train']['label'])

    # evaluate
    return evaluate_bayesian(gnb1.predict(tgidf_test), dataset['test']['label'], 'GaussianNaiveBayes', ver)


def evaluate_bayesian(predicted, actual, name: str, ver: int = 1):
    if ver == 1:
        return {"name": name, "accuracy": accuracy_score(predicted, actual),
                "precision": precision_score(predicted, actual),
                "recall": recall_score(predicted, actual), "f1": f1_score(predicted, actual),
                'type': ver}
    return {"name": name, "accuracy": accuracy_score(predicted, actual),
            "precision": precision_score(predicted, actual, average='macro'),
            "recall": recall_score(predicted, actual, average='macro'),
            "f1": f1_score(predicted, actual, average='macro'),
            'type': ver}


def fasttext_format(name: str, dataset: Dataset):
    items = ['train', 'test']
    for item in items:
        with open(f'{name}_{item}.txt', "w", encoding="utf-8") as f:
            for label, text in zip(dataset[item]['label'], dataset[item]['text']):
                f.write(f"__label__{label} {text}\n")


def fasttext_classifier(filename: str, path: str = '.'):
    model = fasttext.train_supervised(f'{path}/{filename}_train.txt')
    result = model.test(f'{path}/{filename}_test.txt')
    return {"name": "FastText", "accuracy": result[1], "precision": None, "recall": None, "f1": None,
            'type': int(filename[-1])}


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def transformer_classifier(dataset: Dataset, name: str, num_labels: int, id2label: dict, label2id: dict):
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_data = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained('allegro/herbert-base-cased',
                                                               num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=f'results/lab07/var_{num_labels}',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(name)

    evaluate_transformers(model, dataset, tokenizer, compute_metrics)

    return model


def evaluate_task1(predicted, actual):
    return {"accuracy": accuracy_score(predicted, actual), "precision": precision_score(predicted, actual),
            "recall": recall_score(predicted, actual), "f1": f1_score(predicted, actual)}


def evaluate_task2(predicted, actual):
    return {"accuracy": accuracy_score(predicted, actual),
            "precision": precision_score(predicted, actual, average='macro'),
            "recall": recall_score(predicted, actual, average='macro'),
            "f1": f1_score(predicted, actual, average='macro')}


def evaluate_transformers(model, dataset, tokenizer, compute_metrics):
    tokenized_dt = dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    trainer = Trainer(model=model,
                      eval_dataset=tokenized_dt["test"],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    return trainer.evaluate()


def transform_results():
    model01 = {'name': 'AutoModelForSequenceClassification',
               'accuracy': 0.892,
               'precision': 0.6096308296852918,
               'recall': 0.8873462428449641,
               'f1': 0.6490954460386775,
               'type': 1}
    model02 = {'name': 'AutoModelForSequenceClassification',
               'accuracy': 0.867,
               'precision': 0.3363914373088685,
               'recall': 0.6222889556222889,
               'f1': 0.31562271508652207,
               'type': 2}
    return model01, model02


def explain(model, text):
    exp = LimeTextExplainer(class_names=["neutral", "bullying"]) \
        .explain_instance(text, lambda x: predictions(model, x)[0], num_features=10)
    return exp.as_list()


def predictions(model, dataset):
    lime_tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    tokenized = [lime_tokenizer(x, truncation=True) for x in dataset]
    trainer = Trainer(model=model, tokenizer=lime_tokenizer)
    return trainer.predict(tokenized)


def missed_predictions(model_path, dataset, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    pred = predictions(model, dataset['text'])[0]

    pred_labels = np.argmax(pred, axis=1)
    conf_matrix = confusion_matrix(dataset['label'], pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()

    zipped = list(zip(pred_labels, dataset['label']))
    tp = zipped.index((1, 1))
    tn = zipped.index((0, 0))
    fp = zipped.index((1, 0))
    fn = zipped.index((0, 1))

    tp_explained = explain(model, dataset[tp]["text"])
    tn_explained = explain(model, dataset[tn]["text"])
    fp_explained = explain(model, dataset[fp]["text"])
    fn_explained = explain(model, dataset[fn]["text"])

    return [
        {'tp': dataset[tp], 'idx': tp, 'expl': tp_explained},
        {'tn': dataset[tn], 'idx': tn, 'expl': tn_explained},
        {'fp': dataset[fp], 'idx': fp, 'expl': fp_explained},
        {'fn': dataset[fn], 'idx': fn, 'expl': fn_explained},
    ]


if __name__ == '__main__':
    dataset1 = load_dataset("poleval2019_cyberbullying", "task01")
    dataset2 = load_dataset("poleval2019_cyberbullying", "task02")

    results = []

    results.append(bayesian_classifier(dataset1, 1))
    results.append(bayesian_classifier(dataset2, 2))

    fasttext_format('ds1', dataset1)
    fasttext_format('ds2', dataset2)

    results.append(fasttext_classifier('ds1', 'data_lab07'))
    results.append(fasttext_classifier('ds2', 'data_lab07'))

    # id2label = {0: "non-harmful", 1: "harmful"}
    # label2id = {"non-harmful": 0, "harmful": 1}

    # transformer_classifier(dataset1, "roberta-detection-bullying-1", 2, id2label, label2id)
    print("done training one")

    id2label = {0: "non-harmful", 1: "cyberbullying", 2: "hate-speech"}
    label2id = {"non-harmful": 0, "cyberbullying": 1, "hate-speech": 2}

    # transformer_classifier(dataset2, "roberta-detection-bullying-2", 3, id2label, label2id)
    print("done training two")

    results.extend([*transform_results()])

    df = pd.DataFrame(results)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[df['type'] == 1])
        print(df[df['type'] == 2])

    # res = missed_predictions('./models/mod_roberta-detection-bullying-1', dataset1['test'], 2)

    ...
