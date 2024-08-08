# Healthcare Chatbot Project

This project demonstrates the creation of a healthcare chatbot using the `transformers` library. The chatbot is designed to answer common questions related to healthcare services.


## Installation

1. Clone the repository:

git clone https://github.com/yourusername/chatbot-project.git
cd chatbot-project

2. Install the required packages:

pip install -r requirements.txt

## Usage

To train the chatbot model, run:

python train.py


## Project Details

- **Data Preparation**: `data/chatbot_data.py` handles the preparation of input data, now loading from `data/chatbot_dataset.json`.
- **Model Initialization**: `models/chatbot_model.py` initializes the tokenizer and model.
- **Training**: `train.py` contains the main script to train the model using the `Trainer` from `transformers`.


