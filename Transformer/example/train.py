from data import IMDBDataModule
from model import TextClassificationTransformer
import pytorch_lightning as pl

NUM_WORDS = 10000
MAX_SEQ_LEN = 128
EMBEDDING_DIM = 128
BATCH_SIZE = 32
NUM_EPOCHS = 5

imdb_data = IMDBDataModule(
    batch_size=BATCH_SIZE, num_words=NUM_WORDS, max_seq_len=MAX_SEQ_LEN
)
model = TextClassificationTransformer(
    d=EMBEDDING_DIM, max_seq_len=MAX_SEQ_LEN, num_tokens=NUM_WORDS
)
logger = pl.loggers.TensorBoardLogger("tb_logs", name="transformer")
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS, default_root_dir="ckpts", gpus=1, logger=logger
)
trainer.fit(model, imdb_data)
_ = trainer.test(model=model, dataloaders=imdb_data.test_dataloader())
