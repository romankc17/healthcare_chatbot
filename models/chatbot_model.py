# models/chatbot_model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('facebook/blenderbot-400M-distill')
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/blenderbot-400M-distill')
    return tokenizer, model
