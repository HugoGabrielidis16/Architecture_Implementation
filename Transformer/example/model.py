import torch
import torch.nn as nn
from implementation.Transformer import TransformerBlock
import torchmetrics
import sys
from torch.optim import Adam
import pytorch_lightning as pl


class TextClassificationTransformer(pl.LightningModule):
    """
    Stacked Transformer blocks for sequence classification.

    Args:
        d (int): The embedding dimension.
        heads (int): The number of attention heads for each transformer block.
        depth (int): The number of transformer blocks.
        max_seq_len (int): The maximum number of tokens of each sequence.
        num_tokens (int): The vocabulary size.
        num_classes (int): The number of classification classes.
        learning_rate (float): The learning rate for the optimizer.
    """

    def __init__(
        self,
        d: int = 128,
        heads: int = 8,
        depth: int = 6,
        max_seq_len: int = 512,
        num_tokens: int = 30000,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
    ):
        super().__init__()
        # Save arguments in self.hparams.
        self.save_hyperparameters()

        self.num_tokens = num_tokens

        # Embeddings for tokens and position.
        self.token_emb = nn.Embedding(num_tokens, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)

        # The stacked transformer blocks.
        self.transformer_blocks = nn.Sequential(
            *[Transformer.TransformerBlock(d=d, heads=heads) for _ in range(depth)]
        )

        # Mapping of final output sequence to class logits.
        self.classification = nn.Linear(d, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            x (torch.LongTensor): A tensor of shape (b, l) of long values
                representing words in some predetermined vocabulary.

        Returns:
            A tensor of shape (b, c) of logits over the classes
                (c is the number of classes).
        """
        b, l = x.size()
        d = self.hparams.d

        # 1. Generate token embeddings. Shape: [b, l, d].
        # 2. Generate position embeddings. Shape: [b, l, d].
        # 3. Generate final embedding by taking the sum of the two embeddings.
        # ----------------
        tokens = self.token_emb(x)
        positions = self.pos_emb(torch.arange(l).to(self.device)).expand(b, l, d)
        embeddings = tokens + positions
        # ----------------

        # 4. Feed the embedding into the transformer blocks. Shape: [b, l, d].
        # 5. Compute the mean latent vector for each sequence.
        #    The mean is applied over dim=1 (time). Shape: [b, d].
        # 6. Classify. Shape: [b, num_classes].
        # ----------------
        out = self.transformer_blocks(embeddings)
        out = out.mean(dim=1)
        out = self.classification(out)
        # ----------------
        return out

    def configure_optimizers(self):
        """Specify the optimizer used for training."""
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        """A single training step.

        Args:
            batch (List(torch.LongTensor, torch.LongTensor)): The current batch
                inputs x and sentiment predictions y from the train dataloader.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss for the current batch.
        """
        x, y = batch

        # Forward pass.
        logits = self(x)

        # Compute the loss with CrossEntropy.
        loss = self.criterion(logits, y)

        # Log the metrics.
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        self.log("acc", self.accuracy(logits, y), on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """A single test step.

        Pytorch Lightning automatically disable gradients and puts model in eval
        mode.

        Args:
            batch (List(torch.LongTensor, torch.LongTensor)): The current batch
                inputs x and sentiment predictions y from the *test* dataloader.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss for the current batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Log the metrics.
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.accuracy(logits, y), on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """A single validation step.

        Since we don't have a seperate validation set we just use the test
        logic.

        Args:
            batch (List(torch.LongTensor, torch.LongTensor)): The current batch
                inputs x and sentiment predictions y from the *val* dataloader.
            batch_idx (int): The index of the current batch.

        Returns:
            The loss for the current batch.
        """
        return self.test_step(batch, batch_idx)
