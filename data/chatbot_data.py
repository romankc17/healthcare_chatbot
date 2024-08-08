# data/chatbot_data.py

import json
import torch

class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, input_encodings, response_encodings):
        self.input_encodings = input_encodings
        self.response_encodings = response_encodings

    def __len__(self):
        return len(self.input_encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_encodings.input_ids[idx],
            'attention_mask': self.input_encodings.attention_mask[idx],
            'labels': self.response_encodings.input_ids[idx]
        }

def prepare_data(file_path='data/chatbot_dataset.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    inputs = [item['input'] for item in data]
    responses = [item['response'] for item in data]

    return inputs, responses
