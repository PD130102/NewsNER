import numpy as np
import pickle
import pandas as pd
import torch
import json 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import joblib
import re 


# Load the pcikle model and tokenizer NER Transformer Model
model = pickle.load(open("../models/NERTransmodel.pkl", "rb"))
tokenizer = pickle.load(open("../models/NERTranstokenizer.pkl", "rb"))
labels_dict = pickle.load(open("../models/NERTranslabels_dict.pkl", "rb"))

# Load the pcikle model and tokenizer TC Fine Tune Model
max_length = 312
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = pickle.load(open("../models/TC_LLM_model.pkl", "rb"))
tokenizer1 = pickle.load(open("../models/TC_LLM_tokenizer.pkl", "rb"))
labels_ids = pickle.load(open("../models/labels_ids.pkl", "rb"))

# Load the pcikle model and tokenizer TC MNB Model with joblib
model2 = joblib.load("../models/TCNB_model.pkl")

# Load the text summarization fine tuned model and tokenizer
model3 = pickle.load(open("../models/TS_model.pkl", "rb"))
tokenizer3 = pickle.load(open("../models/TS_tokenizer.pkl", "rb"))

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

labels_dict = {
    0: "B-FESTIVAL",
    1: "B-GAME",
    2: "B-LANGUAGE",
    3: "B-LITERATURE",
    4: "B-LOCATION",
    5: "B-MISC",
    6: "B-NUMEX",
    7: "B-ORGANIZATION",
    8: "B-PERSON",
    9: "B-RELIGION",
    10: "B-TIMEX",
    11: "I-FESTIVAL",
    12: "I-GAME",
    13: "I-LANGUAGE",
    14: "I-LITERATURE",
    15: "I-LOCATION",
    16: "I-MISC",
    17: "I-NUMEX",
    18: "I-ORGANIZATION",
    19: "I-PERSON",
    20: "I-RELIGION",
    21: "I-TIMEX",
    22: "O"
}
def text_summarizer(article_text):
    input_ids = tokenizer3([WHITESPACE_HANDLER(article_text)],return_tensors="pt",padding="max_length", truncation=True,max_length=512)["input_ids"]
    output_ids = model3.generate(input_ids=input_ids,max_length=84,no_repeat_ngram_size=2,num_beams=4)[0]
    summary = tokenizer3.decode(output_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return summary

def ner_predict(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_labels = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = predicted_labels.squeeze().tolist()
    predicted_labels = [labels_dict[label] for label in labels]
    result = list(zip(tokens, predicted_labels))
    json_data = [{'term': term, 'label': label} for term, label in result]
    return json.dumps(json_data)

def response_parse(data):
    data = data.encode('utf-8').decode('unicode_escape')
    data = data.replace('"', '')
    data = data.replace('[', '')
    data = data.replace(']', '')
    data = data[1:-1]
    data = data.split('}, {')
    return data

def predict_LLM(sentence):
    encoding = tokenizer1.encode_plus (sentence,
                                        max_length=max_length,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        return_attention_mask=True,
                                        return_token_type_ids=False)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    outputs = model1(input_ids, attention_mask)
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    predict_content = logits.argmax(axis=-1).flatten().tolist()
    predict_label = list(labels_ids.keys())[list(labels_ids.values()).index(predict_content[0])]
    return predict_label

def predict_MNB(text):
    pred = model2.predict([text])
    return pred[0]
