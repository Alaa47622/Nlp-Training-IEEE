import torch
from torch.utils.data import Dataset


class ToxicDataset(Dataset):
    def __init__(self, dataframe, vocab, labels, max_length):
        self.df = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(
            self.vocab.encode(row["comment_text"], self.max_length),
            dtype=torch.long
        )
        y = torch.tensor(row[self.labels].values.astype("float32"))
        return x, y
