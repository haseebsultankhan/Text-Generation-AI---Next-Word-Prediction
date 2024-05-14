# Text Generation AI

This project implements a text generation model using a Long Short-Term Memory (LSTM) neural network architecture in TensorFlow. The model is trained on a dataset of news articles and can generate new text by predicting the next word based on the previous sequence of words.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- NLTK

## Installation

1. Clone the repository:

```bash
git clone https://github.com/haseebsultankhan/text-generation-ai.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage

1. Prepare the dataset by downloading the `fake_or_real_news.csv` file and placing it in the project directory.
2. Run the `Text Generation AI.ipynb` notebook to train the model and generate new text.

## Model Architecture

The model consists of two LSTM layers followed by a Dense layer with a SoftMax activation function. The input to the model is a sequence of tokenized words, and the output is a probability distribution over the entire vocabulary for the next word prediction.

## Training

The training process involves the following steps:

- Load the dataset and preprocess the text.
- Tokenize the text and build a vocabulary.
- Create input sequences and target sequences for training.
- Convert the data into a suitable format for the model.
- Define the model architecture.
- Compile the model with an optimizer and loss function.
- Train the model on the prepared data.

## Text Generation

After training, the model can be used to generate new text by providing a seed text sequence. The `generate_text` function takes the seed text, the desired length of the generated text, and an optional creativity parameter to control the randomness of the predictions.

Example:

```python
generated_text = generate_text("The president of the United States", 100, creativity=5)
print(generated_text)
```


### This will generate a new text sequence of 100 words, starting with the seed text "The president of the United States", with a creativity level of 5.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
