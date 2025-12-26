# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from torchdiffeq import odeint

from sam_audio.model.align import AlignModalities
from sam_audio.model.base import BaseModel
from sam_audio.model.codec import DACVAE
from sam_audio.model.config import SAMAudioConfig
from sam_audio.model.text_encoder import T5TextEncoder
from sam_audio.model.transformer import DiT
from sam_audio.model.vision_encoder import PerceptionEncoder
from sam_audio.processor import Batch
from sam_audio.ranking import create_ranker

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 2 / 32}}


logger = logging.getLogger(__name__)


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _log_cuda_mem(device: torch.device, prefix: str) -> None:
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    logger.info(
        "%s cuda mem: allocated=%s reserved=%s",
        prefix,
        _format_bytes(allocated),
        _format_bytes(reserved),
    )


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


class SAMAudio(BaseModel):
    config_cls = SAMAudioConfig
    revision = None

    def __init__(self, cfg: SAMAudioConfig):
        super().__init__()
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        self.vision_encoder = PerceptionEncoder(cfg.vision_encoder)
        self.transformer = DiT(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self.visual_ranker = create_ranker(cfg.visual_ranker)
        self.text_ranker = create_ranker(cfg.text_ranker)
        if cfg.span_predictor is not None:
            self.span_predictor = PEAudioFrame.from_config(
                cfg.span_predictor, pretrained=True
            )
            self.span_predictor_transform = PEAudioFrameTransform.from_config(
                cfg.span_predictor
            )

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def _module_device(
        self,
        module: Optional[torch.nn.Module],
        default: Optional[torch.device] = None,
    ) -> torch.device:
        if module is None:
            return default or torch.device("cpu")
        for param in module.parameters(recurse=True):
            return param.device
        for buf in module.buffers(recurse=True):
            return buf.device
        return default or torch.device("cpu")

    def _module_dtype(
        self,
        module: Optional[torch.nn.Module],
        default: torch.dtype = torch.float32,
    ) -> torch.dtype:
        if module is None:
            return default
        for param in module.parameters(recurse=True):
            return param.dtype
        for buf in module.buffers(recurse=True):
            return buf.dtype
        return default

    def _main_device(self) -> torch.device:
        return self._module_device(self.transformer, default=torch.device("cpu"))

    def _encoder_device(self) -> torch.device:
        if hasattr(self.audio_codec, "encoder"):
            return self._module_device(self.audio_codec.encoder, self._main_device())
        return self._module_device(self.audio_codec, self._main_device())

    def _decoder_device(self) -> torch.device:
        if hasattr(self.audio_codec, "decoder"):
            return self._module_device(self.audio_codec.decoder, self._main_device())
        return self._module_device(self.audio_codec, self._main_device())

    def _text_device(self) -> torch.device:
        return self._module_device(self.text_encoder, self._main_device())

    def _vision_device(self) -> torch.device:
        return self._module_device(self.vision_encoder, self._main_device())

    def _ranker_device(self) -> torch.device:
        if self.visual_ranker is not None:
            return self._module_device(self.visual_ranker, self._main_device())
        if self.text_ranker is not None:
            return self._module_device(self.text_ranker, self._main_device())
        return self._main_device()

    def _ranker_dtype(self) -> torch.dtype:
        if self.visual_ranker is not None:
            return self._module_dtype(self.visual_ranker)
        if self.text_ranker is not None:
            return self._module_dtype(self.text_ranker)
        return self._module_dtype(self.transformer)

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        return aligned

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            masked_video_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def _get_audio_features(self, audios: torch.Tensor):
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_video_features(
        self,
        video,
        audio_features: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        B, T, _ = audio_features.shape
        if video is None:
            target_device = device or audio_features.device
            return torch.zeros(
                B,
                self.vision_encoder.dim,
                T,
                device=target_device,
                dtype=audio_features.dtype,
            )
        else:
            if device is not None:
                video = [
                    v.to(device) if torch.is_tensor(v) else v for v in video
                ]
            return self.vision_encoder(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _unrepeat_from_reranking(self, tensor, candidates):
        return tensor[::candidates]

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        main_device = self._main_device()
        encoder_device = self._encoder_device()
        audio_features = self._get_audio_features(batch.audios.to(encoder_device))
        if audio_features.device != main_device:
            audio_features = audio_features.to(main_device)

        text_features, text_mask = self.text_encoder(batch.descriptions)
        text_features = text_features.to(
            device=main_device, dtype=audio_features.dtype
        )
        text_mask = text_mask.to(main_device)

        masked_video_features = self._get_video_features(
            batch.masked_video,
            audio_features,
            device=self._vision_device(),
        )
        if masked_video_features.device != main_device:
            masked_video_features = masked_video_features.to(
                device=main_device, dtype=audio_features.dtype
            )

        anchor_ids = batch.anchor_ids.to(main_device)
        anchor_alignment = batch.anchor_alignment.to(main_device)
        audio_pad_mask = (
            batch.audio_pad_mask.to(main_device)
            if batch.audio_pad_mask is not None
            else None
        )

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                audio_pad_mask, candidates
            ),
        }

    def predict_spans(
        self, batch: Batch, audio_features: torch.Tensor, audio_pad_mask: torch.Tensor
    ) -> Batch:
        input = self.span_predictor_transform(text=batch.descriptions).to(
            audio_features.device
        )
        output = self.span_predictor(
            input_features=audio_features[:, :, :128],
            padding_mask=audio_pad_mask,
            return_spans=True,
            **input,
        )
        anchors = [[["+"] + anchor for anchor in anchors] for anchors in output.spans]
        batch.process_anchors(anchors)
        return batch

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
        candidate_batch_size: Optional[int] = None,
        ranker_batch_size: Optional[int] = None,
    ) -> SeparationResult:
        # Encode audio
        forward_args_base = self._get_forward_args(batch, candidates=1)

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=forward_args_base["audio_features"],
                audio_pad_mask=forward_args_base["audio_pad_mask"],
            )

        audio_features_base = forward_args_base["audio_features"]
        bsz, T, C = audio_features_base.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if candidate_batch_size is None or candidate_batch_size <= 0:
            candidate_batch_size = reranking_candidates
        candidate_batch_size = min(candidate_batch_size, reranking_candidates)
        if ranker_batch_size is None or ranker_batch_size <= 0:
            ranker_batch_size = reranking_candidates
        ranker_batch_size = min(ranker_batch_size, reranking_candidates)

        def repeat_args(args, candidates: int):
            repeated = {}
            for key, value in args.items():
                if value is None:
                    repeated[key] = None
                else:
                    repeated[key] = self._repeat_for_reranking(value, candidates)
            return repeated

        target_chunks = [[] for _ in range(bsz)]
        residual_chunks = [[] for _ in range(bsz)]
        noise_chunks = []
        noise_offset = 0

        ranker_device = self._ranker_device()
        ranker_dtype = self._ranker_dtype()
        store_device = ranker_device
        store_dtype = ranker_dtype
        if ranker_device.type == "cuda" and ranker_batch_size < reranking_candidates:
            store_device = torch.device("cpu")
            store_dtype = ranker_dtype
        logger.info(
            "Rerank config: candidates=%d candidate_batch_size=%d ranker_batch_size=%d ranker_device=%s ranker_dtype=%s store_device=%s",
            reranking_candidates,
            candidate_batch_size,
            ranker_batch_size,
            ranker_device,
            ranker_dtype,
            store_device,
        )

        for start in range(0, reranking_candidates, candidate_batch_size):
            cur = min(candidate_batch_size, reranking_candidates - start)
            logger.info(
                "Generating candidates: start=%d count=%d",
                start,
                cur,
            )
            forward_args = repeat_args(forward_args_base, cur)

            if noise is None:
                noise_chunk = torch.randn_like(forward_args["audio_features"])
            else:
                if noise.shape[0] == bsz * reranking_candidates:
                    noise_chunk = noise[noise_offset : noise_offset + bsz * cur]
                    noise_offset += bsz * cur
                elif noise.shape[0] == bsz:
                    noise_chunk = (
                        noise.unsqueeze(1)
                        .expand(bsz, cur, *noise.shape[1:])
                        .reshape(bsz * cur, *noise.shape[1:])
                    )
                else:
                    raise ValueError(
                        "noise must have batch size of bsz or bsz * reranking_candidates"
                    )

            def vector_field(t, noisy_audio):
                res = self.forward(
                    noisy_audio=noisy_audio,
                    time=t.expand(noisy_audio.size(0)),
                    **forward_args,
                )
                return res

            states = odeint(
                vector_field,
                noise_chunk,
                torch.tensor([0.0, 1.0], device=noise_chunk.device),
                **ode_opt,
            )
            generated_features = states[-1].transpose(1, 2)
            # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
            wavs = self.audio_codec.decode(
                generated_features.reshape(2 * bsz * cur, C, T)
            ).view(bsz * cur, 2, -1)

            sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
            if torch.is_tensor(sizes) and sizes.device != wavs.device:
                sizes = sizes.to(wavs.device)
            target_wavs_chunk = self.unbatch(
                wavs[:, 0].view(bsz, cur, -1), sizes
            )
            residual_wavs_chunk = self.unbatch(
                wavs[:, 1].view(bsz, cur, -1), sizes
            )

            for i in range(bsz):
                target_chunks[i].append(
                    target_wavs_chunk[i].to(device=store_device, dtype=store_dtype)
                )
                residual_chunks[i].append(
                    residual_wavs_chunk[i].to(device=store_device, dtype=store_dtype)
                )

            if noise is None:
                noise_chunks.append(noise_chunk)

        if noise is None:
            noise = torch.cat(noise_chunks, dim=0) if noise_chunks else None

        target_wavs = [
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            for chunks in target_chunks
        ]
        residual_wavs = [
            torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
            for chunks in residual_chunks
        ]

        if reranking_candidates > 1 and ranker_device.type == "cuda":
            logger.info(
                "Clearing CUDA cache before reranking on %s",
                ranker_device,
            )
            with torch.cuda.device(ranker_device):
                torch.cuda.empty_cache()
            _log_cuda_mem(ranker_device, "Ranker post empty_cache")

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            videos = batch.masked_video
            if videos is not None and ranker_device.type != "cpu":
                videos = [
                    v.to(ranker_device) if torch.is_tensor(v) else v for v in videos
                ]
            scores_chunks = []
            for start in range(0, reranking_candidates, ranker_batch_size):
                end = min(reranking_candidates, start + ranker_batch_size)
                extracted_chunk = [
                    wav[start:end].to(device=ranker_device, dtype=ranker_dtype)
                    for wav in target_wavs
                ]
                if logger.isEnabledFor(logging.INFO):
                    extracted_len = (
                        extracted_chunk[0].shape[-1] if extracted_chunk else 0
                    )
                    logger.info(
                        "Visual ranker batch: start=%d end=%d extracted_len=%d device=%s dtype=%s",
                        start,
                        end,
                        extracted_len,
                        ranker_device,
                        ranker_dtype,
                    )
                    _log_cuda_mem(ranker_device, "Visual ranker pre")
                scores_chunks.append(
                    self.visual_ranker(
                        extracted_audio=extracted_chunk,
                        videos=videos,
                        sample_rate=self.audio_codec.sample_rate,
                    )
                )
            scores = (
                scores_chunks[0]
                if len(scores_chunks) == 1
                else torch.cat(scores_chunks, dim=1)
            )
            idxs = scores.argmax(dim=1).to("cpu")
        elif reranking_candidates > 1 and self.text_ranker is not None:
            sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
            input_audio = [
                audio[:, : int(size)].cpu().expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores_chunks = []
            for start in range(0, reranking_candidates, ranker_batch_size):
                end = min(reranking_candidates, start + ranker_batch_size)
                extracted_chunk = [
                    wav[start:end].to(device=ranker_device, dtype=ranker_dtype)
                    for wav in target_wavs
                ]
                input_chunk = [
                    audio[start:end].to(device=ranker_device, dtype=ranker_dtype)
                    for audio in input_audio
                ]
                if logger.isEnabledFor(logging.INFO):
                    input_len = input_chunk[0].shape[-1] if input_chunk else 0
                    extracted_len = (
                        extracted_chunk[0].shape[-1] if extracted_chunk else 0
                    )
                    logger.info(
                        "Text ranker batch: start=%d end=%d input_len=%d extracted_len=%d device=%s dtype=%s",
                        start,
                        end,
                        input_len,
                        extracted_len,
                        ranker_device,
                        ranker_dtype,
                    )
                    _log_cuda_mem(ranker_device, "Text ranker pre")
                scores_chunks.append(
                    self.text_ranker(
                        extracted_audio=extracted_chunk,
                        input_audio=input_chunk,
                        descriptions=batch.descriptions,
                        sample_rate=self.audio_codec.sample_rate,
                    )
                )
            scores = (
                scores_chunks[0]
                if len(scores_chunks) == 1
                else torch.cat(scores_chunks, dim=1)
            )
            idxs = scores.argmax(dim=1).to("cpu")
        else:
            idxs = torch.zeros(bsz, dtype=torch.long)

        return SeparationResult(
            target=[
                wav[idx.item()] for wav, idx in zip(target_wavs, idxs, strict=False)
            ],
            residual=[
                wav[idx.item()] for wav, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    def unbatch(self, wavs: torch.Tensor, sizes: torch.Tensor, time_dim: int = -1):
        result = []
        for row, size in zip(wavs, sizes, strict=False):
            result.append(row.narrow(dim=time_dim, start=0, length=size))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile(
                "(^text_encoder|^visual_ranker|^text_ranker|^span_predictor)"
            )
            missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


__all__ = ["SAMAudio"]
