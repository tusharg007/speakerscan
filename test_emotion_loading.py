"""Verify that pretrained emotion head tensors survive model loading."""

import tempfile
import unittest

import torch
from transformers import Wav2Vec2Config

from emotion_model import Wav2Vec2EmotionModel


class EmotionLoadingTests(unittest.TestCase):
    def test_checkpoint_head_is_restored_and_prediction_is_repeatable(self):
        config = Wav2Vec2Config(
            hidden_size=8, intermediate_size=16, num_attention_heads=2,
            num_hidden_layers=1, conv_dim=(4,), conv_stride=(2,), conv_kernel=(3,),
            num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0, final_dropout=0.0,
            id2label={0: "angry", 1: "calm"}, pooling_mode="mean",
        )
        model = Wav2Vec2EmotionModel(config).eval()
        with torch.no_grad():
            model.classifier.output.weight.zero_()
            model.classifier.output.bias.copy_(torch.tensor([-2.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            restored, loading = Wav2Vec2EmotionModel.from_pretrained(
                tmp, output_loading_info=True
            )
        self.assertEqual(loading["missing_keys"], [])
        self.assertEqual(loading["unexpected_keys"], [])
        for name, expected in model.classifier.state_dict().items():
            torch.testing.assert_close(restored.classifier.state_dict()[name], expected)
        with torch.inference_mode():
            first = restored(torch.zeros(1, 80))
            second = restored(torch.zeros(1, 80))
        torch.testing.assert_close(first, torch.tensor([[-2.0, 2.0]]))
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
