
import torch
from transformers import Trainer, TrainingArguments
from data.chatbot_data import prepare_data, ChatbotDataset
from models.chatbot_model import get_model_and_tokenizer

def main():
    inputs, responses = prepare_data()
    tokenizer, model = get_model_and_tokenizer()

    input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    response_encodings = tokenizer(responses, padding=True, truncation=True, return_tensors="pt")

    dataset = ChatbotDataset(input_encodings, response_encodings)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

if __name__ == "__main__":
    main()
