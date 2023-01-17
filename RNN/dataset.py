import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchtext.data import get_tokenizer
from sklearn.model_selection import train_test_split

"""
We are gonna use our RNN to approximate a sin function with some noise.
"""


def load_dataset():
    dataset = pd.read_csv("IMDB Dataset.csv")  # load the dataset
    X = dataset.review
    y = dataset.sentiment.map(
        {"positive": 1, "negative": 0}
    )  # map the labels to 0 and 1
    tokenizer = get_tokenizer("basic_english")

    X = X.map(
        lambda text: tokenizer.encode(
            text, padding="max_length", truncation=True, max_length=64
        )
    )
    X_train, X_test, y_train, y_test = train_test_split(
        torch.Tensor(X), torch.Tensor(y), test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    print(X_train[:2])
