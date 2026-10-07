"""Load the emotion checkpoint's original dense/tanh/output head.

The checkpoint's classifier.dense/output tensors do not match the modern
Transformers sequence-classification head. Loading it through AutoModel
silently initializes a new classifier. This adapter keeps the saved head.
"""

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel


class EmotionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.output = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled):
        hidden = torch.tanh(self.dense(self.dropout(pooled)))
        return self.output(self.dropout(hidden))


class Wav2Vec2EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "pooling_mode", "mean") != "mean":
            raise ValueError("The emotion adapter requires mean pooling")
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = EmotionHead(config)
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        features = self.wav2vec2(
            input_values, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        # Each call contains one unpadded turn, as in the original head.
        return self.classifier(features.mean(dim=1))


class EmotionPredictor:
    """Keep the existing array/sampling_rate -> label/score call interface."""

    def __init__(self, model, feature_extractor, device):
        self.model = model.to(device).eval()
        self.feature_extractor = feature_extractor
        self.device = device

    def __call__(self, audio):
        inputs = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            probabilities = self.model(**inputs).softmax(dim=-1)[0]
        score, index = probabilities.max(dim=0)
        return [{"label": self.model.config.id2label[index.item()], "score": score.item()}]


def load_emotion_predictor(model_id, device):
    model, loading = Wav2Vec2EmotionModel.from_pretrained(
        model_id, output_loading_info=True
    )
    problems = {
        key: loading[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading.get(key)
    }
    if problems:
        raise RuntimeError(f"Emotion checkpoint did not load completely: {problems}")
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    return EmotionPredictor(model, extractor, device)
