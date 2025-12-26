# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import logging

import torch

from ..model.config import JudgeRankerConfig
from ..model.judge import SAMAudioJudgeModel
from ..processor import SAMAudioJudgeProcessor
from .ranker import Ranker


class JudgeRanker(Ranker):
    def __init__(self, config: JudgeRankerConfig):
        super().__init__()
        self.config = config
        self.model = SAMAudioJudgeModel.from_pretrained(config.checkpoint_or_model_id)
        self.processor = SAMAudioJudgeProcessor.from_pretrained(
            config.checkpoint_or_model_id
        )

    @torch.inference_mode()
    def forward(
        self,
        input_audio: list[torch.Tensor],
        extracted_audio: list[torch.Tensor],
        descriptions: list[str],
        sample_rate: int = 48_000,
        **kwargs,
    ):
        logger = logging.getLogger(__name__)
        bsz, ncandidates = len(input_audio), len(input_audio[0])
        if logger.isEnabledFor(logging.INFO):
            input_len = input_audio[0].shape[-1] if input_audio else 0
            extracted_len = extracted_audio[0].shape[-1] if extracted_audio else 0
            device = input_audio[0].device if input_audio else torch.device("cpu")
            dtype = input_audio[0].dtype if input_audio else None
            logger.info(
                "JudgeRanker: bsz=%d ncandidates=%d input_len=%d extracted_len=%d device=%s dtype=%s",
                bsz,
                ncandidates,
                input_len,
                extracted_len,
                device,
                dtype,
            )
        input_seqs = [x[None] for candidates in input_audio for x in candidates]
        extracted_seqs = [x[None] for candidates in extracted_audio for x in candidates]
        repeated_descriptions = [x for x in descriptions for _ in range(ncandidates)]
        processed = self.processor(
            text=repeated_descriptions,
            input_audio=input_seqs,
            separated_audio=extracted_seqs,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate,
        )
        if logger.isEnabledFor(logging.INFO):
            input_values = processed.get("input_values")
            separated_values = processed.get("separated_values")
            logger.info(
                "JudgeRanker processed: input_values=%s separated_values=%s",
                tuple(input_values.shape) if input_values is not None else None,
                tuple(separated_values.shape) if separated_values is not None else None,
            )
        res = self.model(**processed.to(input_audio[0].device))
        return res.overall.view(bsz, ncandidates)
