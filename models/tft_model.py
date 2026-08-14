import torch
import lightning.pytorch as pl
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer as BaseTemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


class TemporalFusionTransformer(BaseTemporalFusionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        return super().from_dataset(dataset, **kwargs)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, weights_only=False, **kwargs):
        return super().load_from_checkpoint(checkpoint_path, weights_only=weights_only, **kwargs)


def create_default_tft_model(training, learning_rate=0.03, hidden_size=64,
                           attention_head_size=4, dropout=0.1,
                           hidden_continuous_size=32, loss=None,
                           log_interval=10, optimizer="ranger",
                           reduce_on_plateau_patience=4):
    if loss is None:
        loss = QuantileLoss()

    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=loss,
        log_interval=log_interval,
        optimizer=optimizer,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
    )


def create_tft_model_with_custom_params(training, custom_params):
    return TemporalFusionTransformer.from_dataset(training, **custom_params)
