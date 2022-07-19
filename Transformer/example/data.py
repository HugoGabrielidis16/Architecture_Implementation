import numpy as np
from sklearn import datasets
import torch
from torch import nn
import torch.nn.functional as F
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class IMDBDataModule(pl.LightningDataModule):
    """
    LightningDataModule to load the IMDB movie review sentiment data.

    Args:
        batch_size (int): The batch size for the train, test, and val
                            dataloader.
        num_words (int): The vocabulary size. The vocabulary is
            sorted by frequency of appearance in the dataset.
        max_seq_len (int): The maximum number of tokens per
            review.
    """

    def __init__(self, batch_size: int, num_words: int, max_seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_words = num_words
        self.max_seq_len = max_seq_len

    def setup(self, *args, **kwargs):
        """
        Initial loading of the dataset and transformation.
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = imdb.load_data(
            num_words=self.num_words, maxlen=self.max_seq_len
        )
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.1
        )
        print(f"# Training Examples: {len(self.y_train)}")
        print(f"# Test Examples: {len(self.y_test)}")

        self.word2idx = dict(
            **{k: v + 3 for k, v in imdb.get_word_index().items()},
            **{
                "<PAD>": 0,
                "<START>": 1,
                "<UNK>": 2,
                "<UNUSED>": 3,
            },
        )
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        # Pad the inputs and convert to torch Tensors.
        self.x_train = pad_sequences(self.x_train, maxlen=self.max_seq_len, value=0.0)
        self.x_test = pad_sequences(self.x_test, maxlen=self.max_seq_len, value=0.0)

    def example(self):
        """Returns a random training example."""
        idx = np.random.randint(0, len(self.x_train))
        x, y = self.x_train[idx], self.y_train[idx]
        review = " ".join(self.idx2word[token_id] for token_id in x if token_id > 1)
        sentiment = "POSITIVE" if y else "NEGATIVE"
        return f"{review}\nSentiment: {sentiment}"

    def train_dataloader(self):
        dataset = TensorDataset(
            torch.LongTensor(self.x_train), torch.LongTensor(self.y_train)
        )
        return DataLoader(dataset, self.batch_size)

    def test_dataloader(self):
        dataset = TensorDataset(
            torch.LongTensor(self.x_test), torch.LongTensor(self.y_test)
        )
        return DataLoader(dataset, self.batch_size)

    def val_dataloader(self):
        dataset = TensorDataset(
            torch.LongTensor(self.x_val), torch.LongTensor(self.y_val)
        )
        return DataLoader(dataset, self.batch_size)


imdb_data = IMDBDataModule(batch_size=128, num_words=30000, max_seq_len=100)
imdb_data.setup()
print("\nExamples:")
print("\n\n".join(imdb_data.example() for _ in range(3)))
