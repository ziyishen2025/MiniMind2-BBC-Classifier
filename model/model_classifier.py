import torch
import torch.nn as nn
import os
from typing import Optional, List, Tuple, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .model_minimind import MiniMindModel, MiniMindConfig


class Classification(PreTrainedModel):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig):
        super().__init__(config)  # 调用父类的初始化方法
        self.config = config

        if not hasattr(config, "num_labels"):
            raise ValueError("MiniMindConfig 必须包含 num_labels 参数。")

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.model = MiniMindModel(config)

        # 2. 分类头 (Classification Head)
        self.score = nn.Sequential(
            # 使用配置中的 dropout 率
            nn.Dropout(p=config.dropout if config.dropout is not None else 0.1),
            # 线性层：将隐藏状态维度 (H) 映射到类别数量 (5)
            nn.Linear(self.hidden_size, self.num_labels, bias=False),
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):

        hidden_states, _, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,  # 分类任务不需要 KV 缓存
            **kwargs
        )

        sequence_output = hidden_states[:, -1, :]

        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs
    ):
        kwargs["ignore_mismatched_sizes"] = True
        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
