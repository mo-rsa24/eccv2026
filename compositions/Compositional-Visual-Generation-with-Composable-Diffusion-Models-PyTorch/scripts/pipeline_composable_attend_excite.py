"""
ComposableAttendAndExcitePipeline
==================================
Combines Logical AND compositional diffusion (Composable Diffusion) with
Attend-and-Excite (A&E) Generative Semantic Nursing (GSN).

Design rationale
----------------
* **AND composition** (Composable Diffusion): at every denoising timestep
  the noise prediction is computed separately for each subject prompt and
  combined via a weighted sum:

      ε̃ = ε_uncond + Σ_i  w_i · (ε_i − ε_uncond)

  This defines a joint score over all subjects but does not prevent the UNet
  from blending them into a single hybrid entity.

* **Attend-and-Excite (GSN)**: during the first `max_iter_to_alter` timesteps
  (the "layout phase") we monitor the 16×16 cross-attention maps for
  user-specified subject tokens in a *joint* prompt (e.g. "a cat and a dog").
  If any token's maximum attention activation is below a target threshold,
  we compute a gradient-based shift of the latent z_t that increases that
  token's peak activation — without changing any model weights.

Together they attack subject-mixture from two complementary angles:
  AND composition ensures each subject's score is respected globally;
  A&E ensures each subject receives a distinct spatial footprint in the
  cross-attention maps.

Requirements
------------
Run inside the `attend_excite` conda environment (diffusers==0.12.1), which
has the `set_attn_processor` API needed for cross-attention hooking.

The Attend-and-Excite repo must be on the Python path so that the utility
modules can be imported.  The runner script handles `sys.path` insertion.
"""

import abc
import inspect
import math
import numbers
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as TF

# ── Import A&E utilities from the sibling repo ───────────────────────────────
_AE_REPO = Path(__file__).resolve().parents[2] / "Attend-and-Excite"
if str(_AE_REPO) not in sys.path:
    sys.path.insert(0, str(_AE_REPO))

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import (
    AttentionStore,
    AttendExciteCrossAttnProcessor,
    aggregate_attention,
)

# ── diffusers imports (pinned to 0.12.1) ─────────────────────────────────────
from diffusers import DDIMScheduler
from diffusers.utils import logging, randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput

logger = logging.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Attention-control registration
# ─────────────────────────────────────────────────────────────────────────────

def _register_attention_control(pipeline, controller: AttentionStore) -> None:
    """
    Replaces every cross- and self-attention processor in `pipeline.unet`
    with an `AttendExciteCrossAttnProcessor` that feeds all attention maps
    into `controller`.

    Must be called once before each generation run so that `controller` is
    freshly bound to the current UNet processors.
    """
    attn_procs: Dict[str, AttendExciteCrossAttnProcessor] = {}
    cross_att_count = 0

    for name in pipeline.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )

    pipeline.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


# ─────────────────────────────────────────────────────────────────────────────
# ComposableAttendAndExcitePipeline
# ─────────────────────────────────────────────────────────────────────────────

class ComposableAttendAndExcitePipeline(StableDiffusionPipeline):
    """
    Text-to-image pipeline that combines:

    1. **Logical AND compositional diffusion** — multi-subject noise prediction
       via per-subject UNet forward passes whose outputs are merged as a
       weighted sum (the AND operator from Composable Diffusion Models).

    2. **Attend-and-Excite GSN** — inference-time gradient-based latent
       correction during the first ``max_iter_to_alter`` timesteps, which
       steers the latent z_t so that every subject token achieves a
       sufficiently high peak cross-attention activation (≥ threshold).

    Parameters
    ----------
    All constructor arguments are inherited from ``StableDiffusionPipeline``.
    """

    # ── A&E helper: encode prompt, return (text_inputs, embeddings) ───────────

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple[Any, torch.FloatTensor]:
        """
        Wraps the parent encoder and additionally returns the raw tokenizer
        output (`text_inputs`) needed by ``_perform_iterative_refinement_step``
        to decode the weakest token name for logging.

        Returns
        -------
        text_inputs : BatchEncoding
            Tokenizer output for the (positive) prompt.
        prompt_embeds : torch.FloatTensor
            Shape ``[2, seq, dim]`` when CFG is active (uncond then cond),
            or ``[1, seq, dim]`` otherwise.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if (
                untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
                and not torch.equal(text_input_ids, untruncated_ids)
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle "
                    f"sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            attn_mask = None
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attn_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attn_mask)[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        bs, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            attn_mask = None
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attn_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=attn_mask
            )[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = (
                negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
                .repeat(1, num_images_per_prompt, 1)
                .view(batch_size * num_images_per_prompt, seq_len, -1)
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    # ── Subject-prompt encoder for AND composition ────────────────────────────

    def _encode_subject_prompts(
        self,
        subject_prompts: List[str],
        device: torch.device,
        num_images_per_prompt: int,
        negative_prompt: Optional[str] = None,
    ) -> torch.FloatTensor:
        """
        Encodes each subject prompt independently and stacks them with a single
        unconditional (empty-string) embedding for classifier-free guidance.

        Returns
        -------
        comp_embeddings : torch.FloatTensor
            Shape ``[1 + N, seq_len, dim]`` where index 0 is the unconditional
            embedding and indices 1..N are the per-subject conditional embeddings.
            N = ``len(subject_prompts)``.

        This layout mirrors the one used by ``ComposableStableDiffusionPipeline``
        for its AND composition loop:
            noise_pred_uncond = noise_preds[:1]
            noise_pred_text   = noise_preds[1:]
        """
        # Unconditional embedding (one shared empty-string encoding)
        _, uncond_emb = self._encode_prompt(
            negative_prompt or "",
            device,
            num_images_per_prompt,
            do_classifier_free_guidance=False,
        )  # shape: [1, seq, dim]

        # Conditional embedding for each subject
        cond_embs = []
        for sp in subject_prompts:
            _, ce = self._encode_prompt(sp, device, num_images_per_prompt, do_classifier_free_guidance=False)
            cond_embs.append(ce)  # each: [1, seq, dim]

        return torch.cat([uncond_emb] + cond_embs, dim=0)  # [1+N, seq, dim]

    # ── A&E loss computation ──────────────────────────────────────────────────

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices_to_alter: List[int],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
    ) -> List[torch.Tensor]:
        """
        For each subject token index, extract the 16×16 spatial attention map,
        optionally apply Gaussian smoothing, and return the maximum activation.

        Gaussian smoothing (kernel_size=3, σ=0.5 by default) prevents a subject
        from collapsing onto a single patch, following the A&E paper.
        """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt if isinstance(self.prompt, str) else self.prompt[0]
            last_idx = len(self.tokenizer(prompt)["input_ids"]) - 1

        # Slice text-token dimension; scale and normalise for comparison
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices because we removed the leading BOS token
        shifted_indices = [idx - 1 for idx in indices_to_alter]

        max_indices_list: List[torch.Tensor] = []
        for i in shifted_indices:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                padded = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
                image = smoothing(padded).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(
        self,
        attention_store: AttentionStore,
        indices_to_alter: List[int],
        attention_res: int = 16,
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
    ) -> List[torch.Tensor]:
        """
        Aggregates cross-attention maps across all UNet layers at `attention_res`
        resolution, then delegates to ``_compute_max_attention_per_index``.
        """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )
        return self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
        )

    @staticmethod
    def _compute_loss(
        max_attention_per_index: List[torch.Tensor],
        return_losses: bool = False,
        loss_mode: str = "max",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """
        GSN loss: for each subject token, penalise any peak activation below 1.

        loss_mode options
        -----------------
        "max"  (default / original A&E):
            L = max_i max(0, 1 − max_attn_i)
            Only the single *worst* token drives the gradient update.
            Conservative — useful when subjects are well-separated.

        "sum":
            L = Σ_i max(0, 1 − max_attn_i)
            Every under-attended token simultaneously contributes to the
            gradient.  More aggressive — recommended when one concept
            (e.g. "dog") is consistently neglected alongside another
            (e.g. "cat") because it enforces *all* subjects in parallel
            rather than round-robining.
        """
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        if loss_mode == "sum":
            loss = sum(losses)
        else:
            loss = max(losses)
        return (loss, losses) if return_losses else loss

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """
        Gradient-based latent shift (no model-weight update):
            z_t ← z_t − step_size · ∇_{z_t} L
        """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        return latents - step_size * grad_cond

    @staticmethod
    def _compute_alpha(
        i: int,
        total_steps: int,
        alpha_start: float,
        alpha_end: float,
        alpha_schedule: str,
    ) -> float:
        """
        Per-timestep blending coefficient α_i for the guided hybrid:

            ε_hybrid = (1 − α_i)·ε_mono + α_i·ε_AND

        Mirrors the ``superdiff_guided`` sampler from trajectory_dynamics_experiment.py
        (Section: ``(1-α)·v_mono + α·v_superdiff``) adapted for the SD1.4
        noise-prediction (ε) paradigm.

        α_start  → strength at the **first** denoising step  (layout phase)
        α_end    → strength at the **last**  denoising step  (detail phase)

        Schedules
        ---------
        "constant" : α_i = α_start  (no decay; original AND composition when = 1.0)
        "linear"   : α linearly decays from α_start to α_end over all timesteps.
                     Recommended default: α_start=1.0, α_end=0.5
                     → first half uses pure AND for spatial separation,
                       second half blends in monolithic for semantic coherence.
        "cosine"   : Smooth S-curve; slower change at extremes, faster in the middle.

        When α=1 at every step the output is identical to the original AND pipeline.
        When α=0 the output is a standard monolithic CFG image.
        """
        if alpha_schedule == "constant" or alpha_start == alpha_end:
            return alpha_start
        t = i / max(total_steps - 1, 1)          # ∈ [0, 1]
        if alpha_schedule == "cosine":
            t = (1.0 - math.cos(math.pi * t)) / 2.0   # S-curve remap
        return alpha_start + t * (alpha_end - alpha_start)

    # ── SuperDIFF kappa solver ─────────────────────────────────────────────────

    @staticmethod
    def _solve_kappa_and(
        velocities: List[torch.Tensor],
        vel_uncond: torch.Tensor,
        dsigma: float,
        sigma: float,
        guidance_scale: float,
        lift: float = 0.0,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """
        Compute SuperDIFF kappa weights for AND composition (Proposition 6).

        Finds κ = [κ₁, …, κₘ] with Σκ = 1 such that every subject's
        log-density evolves at the same rate under the composite update:

            dx = 2·dσ · (ε_u + gs · Σₖ κₖ · (εₖ − ε_u))

        For a deterministic scheduler (DDIM) the stochastic noise term is
        zero, so dx_base = 2·dσ·ε_u.

        Ported from notebooks/dynamics.py::_solve_kappa_and (SD1.4, Euler).
        """
        M = len(velocities)
        B = velocities[0].shape[0]
        dev = velocities[0].device

        # Flatten spatial dims → [M, B, D]
        vels  = torch.stack([v.flatten(1) for v in velocities])   # [M, B, D]
        v_unc = vel_uncond.flatten(1)                              # [B, D]

        # dx_base for deterministic inference (noise = 0)
        dx_base = (2.0 * dsigma * vel_uncond).flatten(1)          # [B, D]

        u_diff = vels - v_unc.unsqueeze(0)                        # [M, B, D]
        v_diff = vels[1:] - vels[0:1]                             # [M-1, B, D]

        u_diff_t = u_diff.permute(1, 0, 2).float()                # [B, M, D]
        v_diff_t = v_diff.permute(1, 0, 2).float()                # [B, M-1, D]

        # Build A  [B, M, M]
        A = torch.zeros(B, M, M, device=dev, dtype=torch.float32)
        A[:, :M - 1, :] = (2.0 * dsigma * guidance_scale) * torch.bmm(
            v_diff_t, u_diff_t.transpose(1, 2)
        )
        A[:, M - 1, :] = 1.0                                      # Σκ = 1

        # Build b  [B, M]
        b = torch.zeros(B, M, device=dev, dtype=torch.float32)
        norms_sq = (vels.float() ** 2).sum(dim=2)                 # [M, B]
        for j in range(M - 1):
            norm_term = abs(dsigma) * (norms_sq[0] - norms_sq[j + 1])
            dot_term  = (dx_base.float() * v_diff_t[:, j, :]).sum(dim=1)
            b[:, j]   = norm_term - dot_term - sigma * lift / num_inference_steps
        b[:, M - 1] = 1.0

        kappa = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)  # [B, M]
        return kappa.to(velocities[0].dtype)

    def _get_sigma_and_dsigma(
        self, i: int, timesteps: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Return (sigma_i, dsigma_i) for denoising step i.

        Supports both EulerDiscreteScheduler (.sigmas attribute) and
        DDIMScheduler (derives sigma from alphas_cumprod).
        """
        if hasattr(self.scheduler, "sigmas"):
            sigma      = float(self.scheduler.sigmas[i])
            sigma_next = float(self.scheduler.sigmas[i + 1]) \
                         if i + 1 < len(self.scheduler.sigmas) else 0.0
        else:
            # DDIM: σ_t = sqrt((1 − ᾱ_t) / ᾱ_t)
            def _alpha_to_sigma(t_int: int) -> float:
                ap = self.scheduler.alphas_cumprod[t_int]
                return float(((1.0 - ap) / ap).sqrt())

            sigma = _alpha_to_sigma(int(timesteps[i].item()))
            sigma_next = (
                _alpha_to_sigma(int(timesteps[i + 1].item()))
                if i + 1 < len(timesteps) else 0.0
            )

        return sigma, sigma_next - sigma   # (sigma, dsigma)

    def _perform_iterative_refinement_step(
        self,
        latents: torch.Tensor,
        indices_to_alter: List[int],
        loss: torch.Tensor,
        threshold: float,
        joint_text_embeddings: torch.Tensor,
        text_input,
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 16,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_refinement_steps: int = 20,
        normalize_eot: bool = False,
        loss_mode: str = "max",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Iterative latent refinement at threshold timesteps.

        Repeatedly updates z_t via the A&E gradient until *every* subject token
        achieves a peak attention ≥ `threshold`, or `max_refinement_steps` is
        exceeded.  Uses only the *joint* conditional embedding for UNet forward
        passes so that the AND composition embeddings are not needed here.
        """
        iteration = 0
        target_loss = max(0, 1.0 - threshold)

        # joint_text_embeddings layout: [uncond, joint_cond]
        joint_cond_emb = joint_text_embeddings[1].unsqueeze(0)

        while loss > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            # Forward pass to capture attention maps for the updated latent
            self.unet(latents, t, encoder_hidden_states=joint_cond_emb).sample
            self.unet.zero_grad()

            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot,
            )
            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True, loss_mode=loss_mode)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            try:
                low_token = np.argmax([l.item() if not isinstance(l, int) else l for l in losses])
            except Exception as e:
                print(e)
                low_token = np.argmax(losses)

            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            print(f"\t Refinement {iteration}: '{low_word}' max-attn = {max_attention_per_index[low_token]:.4f}")

            if iteration >= max_refinement_steps:
                print(f"\t Exceeded max refinement steps ({max_refinement_steps}).")
                break

        # Final forward pass to get the definitive attention maps / loss value
        latents = latents.clone().detach().requires_grad_(True)
        self.unet(latents, t, encoder_hidden_states=joint_cond_emb).sample
        self.unet.zero_grad()

        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
        )
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True, loss_mode=loss_mode)
        print(f"\t Refinement finished. Final loss: {loss:.4f}")
        return loss, latents, max_attention_per_index

    # ── Main generation call ──────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        # ── Joint prompt (for A&E attention monitoring) ───────────────────────
        prompt: Union[str, List[str]],
        # ── Subject prompts (for AND composition) ─────────────────────────────
        subject_prompts: List[str],
        # ── Tokens to enforce via A&E ─────────────────────────────────────────
        indices_to_alter: List[int],
        # ── AND composition weights ───────────────────────────────────────────
        comp_weights: Optional[Union[str, List[float]]] = None,
        # ── Standard generation params ────────────────────────────────────────
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # ── A&E layout-phase params ───────────────────────────────────────────
        attention_res: int = 16,
        max_iter_to_alter: int = 25,
        run_standard_sd: bool = False,
        thresholds: Optional[Dict[int, float]] = None,
        scale_factor: int = 20,
        scale_range: Tuple[float, float] = (1.0, 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        sd_2_1: bool = False,
        # ── Guided hybrid: α-blend between AND and monolithic ─────────────────
        alpha: float = 1.0,
        alpha_end: float = 0.5,
        alpha_schedule: str = "linear",
        # ── A&E loss aggregation mode ─────────────────────────────────────────
        loss_mode: str = "sum",
    ):
        """
        Generate an image from multiple subject prompts using Logical AND
        composition, guided by Attend-and-Excite GSN during the layout phase.

        Parameters
        ----------
        prompt : str
            The *joint* prompt used for A&E attention monitoring.
            Should contain all subjects in natural language, e.g.
            ``"a cat and a dog"``.
        subject_prompts : List[str]
            Individual subject descriptions for AND composition,
            e.g. ``["a cat", "a dog"]``.
        indices_to_alter : List[int]
            Token positions (1-indexed, BOS excluded) in `prompt` that
            correspond to the subject nouns.  Use the tokenizer to find them.
        comp_weights : str | List[float] | None
            Per-subject guidance scale for AND composition.  If ``None``,
            all subjects share ``guidance_scale``.  May also be provided as a
            pipe-separated string ``"7.5 | 7.5"``.
        max_iter_to_alter : int
            Number of denoising timesteps (from the start) during which A&E
            gradient corrections are applied.  Default: 25 (layout phase only).
        thresholds : dict {timestep_index: float}
            At these specific steps an iterative refinement inner-loop is run
            until the minimum peak attention exceeds the threshold value.
            Default: ``{0: 0.05, 10: 0.5, 20: 0.8}``.
        scale_factor : int
            Base gradient step size for the A&E latent update.
        scale_range : (float, float)
            Linear decay range for the gradient step size over the layout phase.
        smooth_attentions : bool
            Apply Gaussian smoothing (kernel_size, sigma) to attention maps.
        sd_2_1 : bool
            Set True when using ``stabilityai/stable-diffusion-2-1-base``
            (adjusts EOT normalisation inside A&E).
        alpha : float
            Starting α for the guided-blend schedule (step 0).
            ``α=1.0`` → pure AND composition; ``α=0.0`` → pure monolithic CFG.
            Matches the ``(1-α)·v_mono + α·v_AND`` formula from the
            trajectory_dynamics_experiment.py ``superdiff_guided`` sampler,
            adapted for SD1.4 noise-prediction.
            Default 1.0 (no blending at start — maximum spatial separation).
        alpha_end : float
            Final α at the last denoising step.  Default 0.5 → by the end
            the noise prediction is 50 % monolithic (semantic coherence)
            + 50 % AND (subject separation).
        alpha_schedule : str
            How α decays from ``alpha`` to ``alpha_end`` over the trajectory.
            ``"constant"`` — fixed α_start throughout (original AND when = 1.0).
            ``"linear"``   — straight-line decay (recommended default).
            ``"cosine"``   — smooth S-curve, slower change at step boundaries.
        loss_mode : str
            How per-token A&E losses are aggregated into a single scalar.
            ``"max"``  — original A&E: only the single worst token drives grad.
            ``"sum"``  — all under-attended tokens contribute simultaneously.
            Use ``"sum"`` (default) when one concept (e.g. "dog") is consistently
            weaker; it enforces both subjects in every gradient step.
        """
        if thresholds is None:
            thresholds = {0: 0.05, 10: 0.5, 20: 0.8}

        # ── 0. Setup ──────────────────────────────────────────────────────────
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps, negative_prompt)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        device = self._execution_device
        do_cfg  = guidance_scale > 1.0

        # Store joint prompt for EOT normalisation inside _compute_max_attention_per_index
        self.prompt = prompt

        # ── 1. Parse composition weights ──────────────────────────────────────
        n_subjects = len(subject_prompts)
        if comp_weights is None:
            cw = torch.tensor([guidance_scale] * n_subjects, device=device).reshape(-1, 1, 1, 1)
        elif isinstance(comp_weights, str):
            cw = torch.tensor(
                [float(w.strip()) for w in comp_weights.split("|")][:n_subjects],
                device=device,
            ).reshape(-1, 1, 1, 1)
        else:
            cw = torch.tensor(comp_weights, device=device).reshape(-1, 1, 1, 1)

        # ── 2. Encode joint prompt for A&E ────────────────────────────────────
        # joint_text_embeddings: [uncond, joint_cond]  shape [2, seq, dim]
        text_inputs, joint_text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_cfg, negative_prompt,
        )
        # Conditional embedding for A&E UNet forward passes
        joint_cond_emb = joint_text_embeddings[1].unsqueeze(0)  # [1, seq, dim]

        # ── 3. Encode subject prompts for AND composition ─────────────────────
        # comp_embeddings: [uncond, cond_1, ..., cond_N]  shape [1+N, seq, dim]
        comp_embeddings = self._encode_subject_prompts(
            subject_prompts, device, num_images_per_prompt, negative_prompt
        )

        # ── 4. Create AttentionStore and register processors ──────────────────
        attention_store = AttentionStore()
        _register_attention_control(self, attention_store)

        # ── 5. Prepare timesteps ──────────────────────────────────────────────
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ── 6. Initialise latents ─────────────────────────────────────────────
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height, width,
            joint_text_embeddings.dtype,
            device, generator, latents,
        )

        # ── 7. Misc setup ─────────────────────────────────────────────────────
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Linearly decay the A&E gradient step size over the layout phase
        scale_range_arr = np.linspace(scale_range[0], scale_range[1], len(timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(timesteps) + 1

        # ── 8. Denoising loop ─────────────────────────────────────────────────
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # ── STAGE A: Attend-and-Excite latent correction ──────────────
                # Applied only during the first `max_iter_to_alter` timesteps.
                # Uses the *joint* prompt embedding so that token-level attention
                # maps correspond to individual subject tokens.

                if not run_standard_sd and i < max_iter_to_alter:
                    with torch.enable_grad():
                        latents = latents.clone().detach().requires_grad_(True)

                        # Forward pass: captures cross-attention into `attention_store`
                        self.unet(
                            latents, t,
                            encoder_hidden_states=joint_cond_emb,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()

                        # Compute peak attention per subject token
                        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                            attention_store=attention_store,
                            indices_to_alter=indices_to_alter,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            normalize_eot=sd_2_1,
                        )

                        loss = self._compute_loss(max_attention_per_index, loss_mode=loss_mode)

                        # Inner iterative-refinement loop at threshold timesteps
                        if i in thresholds and loss > 1.0 - thresholds[i]:
                            torch.cuda.empty_cache()
                            loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latents,
                                indices_to_alter=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                joint_text_embeddings=joint_text_embeddings,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * float(np.sqrt(scale_range_arr[i])),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1,
                                loss_mode=loss_mode,
                            )

                        # Gradient-based latent shift for the current step
                        if loss != 0:
                            latents = self._update_latent(
                                latents=latents,
                                loss=loss,
                                step_size=scale_factor * float(np.sqrt(scale_range_arr[i])),
                            )

                        print(f"Step {i:>3d} | A&E loss: {loss:.4f}")

                # ── STAGE B: AND Composition + Guided Alpha Blend ─────────────
                # Computes the Logical AND noise prediction and optionally blends
                # it with a monolithic CFG prediction, following the guided hybrid
                # from trajectory_dynamics_experiment.py:
                #
                #   ε_AND  = ε_u + Σ_i  w_i · (ε_i − ε_u)        [AND operator]
                #   ε_mono = ε_u + gs  · (ε_joint − ε_u)          [monolithic CFG]
                #   ε_out  = (1 − α_t)·ε_mono + α_t·ε_AND         [guided blend]
                #
                # α=1 → pure AND (spatial separation); α=0 → pure monolithic.
                # The schedule decays α over time: AND-dominant in the layout
                # phase, monolithic-dominant in the texture/detail phase.

                # Scale the latent for this timestep
                latent_input = self.scheduler.scale_model_input(latents, t)

                # Per-component noise predictions  [1+N, C, H, W]
                noise_preds: List[torch.Tensor] = []
                for j in range(comp_embeddings.shape[0]):
                    noise_preds.append(
                        self.unet(
                            latent_input,
                            t,
                            encoder_hidden_states=comp_embeddings[j: j + 1],
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                    )
                noise_pred_stack = torch.cat(noise_preds, dim=0)  # [1+N, C, H, W]

                # AND guidance:  ε_AND = ε_u + Σ_i  w_i · (ε_i − ε_u)
                noise_pred_uncond = noise_pred_stack[:1]   # [1, C, H, W]
                if do_cfg:
                    noise_pred_text = noise_pred_stack[1:]  # [N, C, H, W]
                    noise_and = noise_pred_uncond + (
                        cw * (noise_pred_text - noise_pred_uncond)
                    ).sum(dim=0, keepdim=True)
                else:
                    noise_and = noise_pred_stack

                # Compute α for this timestep
                alpha_t = self._compute_alpha(
                    i, len(timesteps), alpha, alpha_end, alpha_schedule
                )

                if do_cfg and alpha_t < 1.0:
                    # Monolithic CFG: ε_mono = ε_u + gs · (ε_joint − ε_u)
                    # Uses the joint-prompt conditional embedding already encoded
                    # for A&E, so no extra tokenization is needed.
                    noise_joint_cond = self.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=joint_cond_emb,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    noise_mono = noise_pred_uncond + guidance_scale * (
                        noise_joint_cond - noise_pred_uncond
                    )
                    # Guided blend: (1-α)·ε_mono + α·ε_AND
                    noise_pred = (1.0 - alpha_t) * noise_mono + alpha_t * noise_and
                else:
                    noise_pred = noise_and

                # Scheduler step: z_{t-1} ← scheduler(ε̃, t, z_t)
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # Callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # ── 9. Decode & return ────────────────────────────────────────────────
        image = self.decode_latents(latents)
        image, has_nsfw = self.run_safety_checker(image, device, joint_text_embeddings.dtype)

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return image, has_nsfw

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw)
