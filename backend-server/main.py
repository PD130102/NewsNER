from typing import Union
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import utils
import json
app = FastAPI(title='API', description='NLP Project API', version='1.0.0')
@app.get("/")
def read_root():
    return {"Welcome to NLP Project API - We have all the different NLP models in one place for you to use!"}

@app.get("/function/NER")
def NER():
    return {json.dumps(utils.ner_predict("अकबर ईद पर टेनिस खेलता है"))}


class SentenceInput(BaseModel):
    sentence: str

@app.post("/get_ner_tags")
async def get_ner_tags_endpoint(input_data: SentenceInput):
    try:
        ner_tags = utils.ner_predict(input_data.sentence)
        print(type(ner_tags))
        output = utils.response_parse(ner_tags)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get_text_classification/LLM")
async def get_text_classification_endpoint(input_data: SentenceInput):
    try:
        print(input_data.sentence)
        text_classification = utils.predict_LLM(input_data.sentence)
        print(text_classification)
        return text_classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get_text_classification/MNB")
async def get_text_classification_endpoint(input_data: SentenceInput):
    try:
        print(input_data.sentence)
        text_classification = utils.predict_MNB(input_data.sentence)
        print(text_classification)
        return text_classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/get_text_summarization")
async def get_text_summarization_endpoint(input_data: SentenceInput):
    try:
        print(input_data.sentence)
        text_summarization = utils.text_summarizer(input_data.sentence)
        print(text_summarization)
        return text_summarization
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))