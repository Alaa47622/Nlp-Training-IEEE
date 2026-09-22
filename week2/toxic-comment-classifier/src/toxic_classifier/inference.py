import torch

from toxic_classifier.config import load_config, path
from toxic_classifier.features.vocab import Vocabulary
from toxic_classifier.models.lstm import ToxicLSTM


class Predictor:
    def __init__(self):
        cfg = load_config()
        self.labels = cfg["data"]["labels"]
        self.max_length = cfg["data"]["max_length"]
        self.vocab = Vocabulary.load(path("data/processed/vocab.json"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ToxicLSTM(len(self.vocab), cfg["model"]["embedding_dim"],
            cfg["model"]["hidden_dim"], cfg["model"]["num_layers"],
            cfg["model"]["dropout"], cfg["model"]["bidirectional"], len(self.labels)).to(self.device)
        self.model.load_state_dict(torch.load(path("models/model.pt"), map_location=self.device))
        self.model.eval()

    def predict(self, text: str):
        x = torch.tensor([self.vocab.encode(text, self.max_length)], dtype=torch.long).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x))[0].cpu().tolist()
        return dict(zip(self.labels, probs))
