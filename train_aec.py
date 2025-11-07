import os
import csv
import math
import random
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, ExponentialLR
from torch.optim.optimizer import Optimizer

# wav IO
import soundfile as sf
import librosa

# PESQ for speech quality evaluation
from pesq import pesq

from stream.deepvqe_aec import DeepVQE_AEC
from stream.loss_factory import HybridLoss


# =====================
# SophiaG Optimizer Implementation
# =====================
class SophiaG(Optimizer):
    """
    SophiaG optimizer implementation based on the paper:
    "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 2e-4)
        betas: coefficients used for computing running averages of gradient and hessian (default: (0.965, 0.99))
        rho: the coefficient used for computing the Hessian-based update (default: 0.04)
        weight_decay: weight decay coefficient (default: 1e-1)
        eps: term added to the denominator to improve numerical stability (default: 1e-12)
        k: the interval for updating the Hessian estimate (default: 10)
    """
    def __init__(self, params, lr=2e-4, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1, eps=1e-12, k=10):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho value: {rho}")

        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay, eps=eps, k=k)
        super(SophiaG, self).__init__(params, defaults)

    def step(self, closure=None, bs=5120):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
            bs: batch size for computing the Hessian estimate (default: 5120)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['hessian'] = torch.zeros_like(p.data)

                exp_avg, hessian = state['exp_avg'], state['hessian']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update Hessian estimate every k steps
                if state['step'] % group['k'] == 1:
                    # Simplified Hessian estimate using gradient squared (Gauss-Newton approximation)
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Corrected estimates
                exp_avg_corrected = exp_avg / bias_correction1
                hessian_corrected = hessian / bias_correction2

                # Compute update
                update = exp_avg_corrected / (group['rho'] * hessian_corrected + group['eps'])
                
                # Clipping
                update = torch.clamp(update, -1.0, 1.0)

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])

                # Apply update
                p.data.add_(update, alpha=-group['lr'])

        return loss


# =====================
# Learning Rate Scheduler with Warmup
# =====================
class WarmupScheduler:
    """Warmup learning rate scheduler that gradually increases learning rate during warmup period."""
    
    def __init__(self, optimizer, warmup_steps, base_scheduler=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, metrics=None):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase: linearly increase learning rate
            warmup_factor = self.step_count / self.warmup_steps
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # After warmup, use base scheduler if provided
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metrics is not None:
                        self.base_scheduler.step(metrics)
                else:
                    self.base_scheduler.step()

    def state_dict(self):
        """Return state for checkpointing.
        Includes warmup progress and underlying scheduler state (if any).
        """
        state = {
            'step_count': self.step_count,
            'warmup_steps': self.warmup_steps,
            'base_lrs': self.base_lrs,
        }
        # Save base scheduler state if present
        if self.base_scheduler is not None:
            try:
                state['base_scheduler'] = self.base_scheduler.state_dict()
            except Exception:
                state['base_scheduler'] = None
        else:
            state['base_scheduler'] = None
        return state

    def load_state_dict(self, state):
        """Load state from checkpoint.
        Restores warmup progress and underlying scheduler state (if any).
        """
        if not isinstance(state, dict):
            return
        self.step_count = int(state.get('step_count', self.step_count))
        self.warmup_steps = int(state.get('warmup_steps', self.warmup_steps))
        base_lrs = state.get('base_lrs', None)
        if base_lrs is not None:
            self.base_lrs = list(base_lrs)
        bs_state = state.get('base_scheduler', None)
        if bs_state is not None and self.base_scheduler is not None:
            try:
                self.base_scheduler.load_state_dict(bs_state)
            except Exception:
                pass


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


# =====================
# STFT/ISTFT utilities
# =====================
class STFTConfig:
    def __init__(self, n_fft=512, hop_length=256, win_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def window(self, device):
        return torch.hann_window(self.win_length, device=device)


def wav_to_stft(x: np.ndarray, cfg: STFTConfig, device: torch.device) -> torch.Tensor:
    """x: (n_samples,), float32, mono 16k -> returns (F=257, T, 2) real/imag"""
    t = torch.from_numpy(x).to(device)
    t = t.unsqueeze(0)  # (B=1, n)
    X = torch.stft(
        t,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=cfg.window(device),
        return_complex=True,
        center=True,
    )  # (1, F, T)
    X = torch.view_as_real(X).squeeze(0)  # (F, T, 2)
    return X


def spec_to_time(spec: torch.Tensor, n_fft: int = 512, hop_length: int = 256, win_length: int = 512) -> torch.Tensor:
    """Convert complex spectrogram to time domain signal.
    
    Args:
        spec: (B, F, T, 2) complex spectrogram with real/imag parts
        n_fft: FFT size
        hop_length: hop length for STFT
        win_length: window length for STFT
        
    Returns:
        time_sig: (B, n_samples) time domain signal
    """
    # Convert (B, F, T, 2) to (B, F, T) complex
    complex_spec = torch.complex(spec[..., 0], spec[..., 1])
    
    # Create window
    window = torch.hann_window(win_length, device=spec.device)
    
    # ISTFT to get time domain signal
    time_sig = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=spec.shape[2] * hop_length  # Approximate length
    )
    
    return time_sig


def pad_or_crop_frames(X: torch.Tensor, T_target: int) -> torch.Tensor:
    """X: (F, T, 2) -> (F, T_target, 2) contiguous segment, pad end with zeros if too short"""
    F, T, C = X.shape
    if T >= T_target:
        start = random.randint(0, T - T_target)
        return X[:, start : start + T_target, :]
    else:
        pad = torch.zeros(F, T_target - T, C, device=X.device, dtype=X.dtype)
        return torch.cat([X, pad], dim=1)

def right_pad_frames(X: torch.Tensor, T_target: int) -> torch.Tensor:
    """Right-pad frames to T_target without cropping. X: (F, T, 2) -> (F, T_target, 2).
    If T >= T_target, returns X unchanged.
    """
    F, T, C = X.shape
    if T >= T_target:
        return X
    pad = torch.zeros(F, T_target - T, C, device=X.device, dtype=X.dtype)
    return torch.cat([X, pad], dim=1)

    


def calculate_pesq(y_pred: torch.Tensor, y_true: torch.Tensor, fs: int = 16000, valid_lengths_time: List[int] | None = None) -> float:
    """
    Calculate PESQ score between predicted and true audio signals
    
    Args:
        y_pred: Predicted audio signal tensor (batch_size, time_samples)
        y_true: True/clean audio signal tensor (batch_size, time_samples)
        fs: Sample rate (default: 16000 Hz)
    
    Returns:
        Average PESQ score across the batch
    """
    pesq_scores = []
    
    # Convert tensors to numpy and process each sample in the batch
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    # Handle single sample case
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np[np.newaxis, :]
        y_true_np = y_true_np[np.newaxis, :]
    
    for i in range(y_pred_np.shape[0]):
        try:
            pred_audio = y_pred_np[i]
            true_audio = y_true_np[i]
            # Trim to valid lengths if provided (in time samples)
            if valid_lengths_time is not None:
                max_len = int(valid_lengths_time[i])
                pred_audio = pred_audio[:max_len]
                true_audio = true_audio[:max_len]
            
            # Check if audio signals have sufficient energy
            pred_energy = np.mean(pred_audio ** 2)
            true_energy = np.mean(true_audio ** 2)
            
            # Skip if either signal is too quiet (likely silence)
            if pred_energy < 1e-6 or true_energy < 1e-6:
                # Use a neutral PESQ score for silent segments
                pesq_scores.append(2.5)  # Neutral score between 1-5 range
                continue
            
            # Normalize audio to appropriate range for PESQ
            # PESQ expects signals in range roughly [-1, 1] but not too quiet
            pred_max = np.max(np.abs(pred_audio))
            true_max = np.max(np.abs(true_audio))
            
            if pred_max > 1e-6:
                pred_audio = pred_audio / pred_max * 0.8  # Scale to 80% of max range
            if true_max > 1e-6:
                true_audio = true_audio / true_max * 0.8
            
            # Ensure minimum length for PESQ (at least 0.25 seconds)
            min_length = int(0.25 * fs)  # 0.25 seconds
            if len(pred_audio) < min_length:
                # Pad with zeros if too short
                pred_audio = np.pad(pred_audio, (0, min_length - len(pred_audio)), 'constant')
                true_audio = np.pad(true_audio, (0, min_length - len(true_audio)), 'constant')
            
            # Calculate PESQ (using wideband mode for 16kHz)
            if fs == 16000:
                pesq_score = pesq(fs, true_audio, pred_audio, 'wb')  # wideband
            else:
                pesq_score = pesq(fs, true_audio, pred_audio, 'nb')  # narrowband
            
            # Clamp PESQ score to valid range [1.0, 4.5]
            pesq_score = np.clip(pesq_score, 1.0, 4.5)
            pesq_scores.append(pesq_score)
            
        except Exception as e:
            # Only print warning for unexpected errors, not for "No utterances detected"
            if "No utterances detected" not in str(e):
                print(f"Warning: PESQ calculation failed for sample {i}: {e}")
            # Use a neutral score for failed calculations
            pesq_scores.append(2.5)
    
    return np.mean(pesq_scores) if pesq_scores else 2.5


def calculate_erle(mic_time: torch.Tensor, processed_time: torch.Tensor, eps: float = 1e-12, align: bool = True, valid_lengths_time: List[int] | None = None) -> float:
    """Calculate ERLE (Echo Return Loss Enhancement) in dB per batch and return batch mean.
    
    ERLE[dB] = 10*log10( E[mic^2] / E[(mic - processed)^2] ).
    为了减少幅度失配带来的偏差，默认对 processed 做最小二乘幅度对齐。
    """
    # Ensure shapes: (B, T)
    if mic_time.ndim == 1:
        mic_time = mic_time.unsqueeze(0)
    if processed_time.ndim == 1:
        processed_time = processed_time.unsqueeze(0)

    # If valid lengths provided, compute per-sample ERLE with trimming
    if valid_lengths_time is not None:
        erle_vals = []
        B = mic_time.shape[0]
        for i in range(B):
            tlen = int(valid_lengths_time[i])
            mt = mic_time[i, :tlen]
            pt = processed_time[i, :tlen]
            if align:
                scale = torch.sum(mt * pt) / (torch.sum(pt * pt) + eps)
                pt = scale * pt
            residual = mt - pt
            num = torch.mean(mt * mt)
            den = torch.mean(residual * residual) + eps
            erle_i = 10.0 * torch.log10((num + eps) / den)
            erle_vals.append(erle_i.item())
        return float(np.mean(erle_vals)) if erle_vals else 0.0
    else:
        # Amplitude alignment (least-squares scaling), optional
        if align:
            scale = torch.sum(mic_time * processed_time, dim=-1, keepdim=True) / (torch.sum(processed_time * processed_time, dim=-1, keepdim=True) + eps)
            processed_time = scale * processed_time

        residual = mic_time - processed_time
        num = torch.mean(mic_time * mic_time, dim=-1)
        den = torch.mean(residual * residual, dim=-1) + eps
        erle = 10.0 * torch.log10((num + eps) / den)
        return erle.mean().item()


# =====================
# Dataset
# =====================
class AECDataset(Dataset):
    """Reads a CSV manifest with columns: mix_filepath,farnerd_filepath,target_filepath,split (paths to wav files).
    Assumes 16kHz mono. Produces STFT segments of fixed frame length.
    """

    def __init__(self, manifest_csv: str, split: str = "train", segment_frames: int = 63, stft_cfg: STFTConfig = STFTConfig(), device: torch.device = torch.device("cpu"), no_crop: bool = False):
        self.items: List[Tuple[str, str, str]] = []
        with open(manifest_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append((row["mix_filepath"], row["farnerd_filepath"], row["target_filepath"]))
        self.segment_frames = segment_frames
        self.stft_cfg = stft_cfg
        self.device = device
        self.no_crop = no_crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        mic_path, far_path, clean_path = self.items[idx]
        mic, fs_mic = sf.read(mic_path, dtype="float32")
        far, fs_far = sf.read(far_path, dtype="float32")
        clean, fs_clean = sf.read(clean_path, dtype="float32")
        
        # 自动重采样到16kHz
        if fs_mic != 16000:
            mic = librosa.resample(mic, orig_sr=fs_mic, target_sr=16000)
        if fs_far != 16000:
            far = librosa.resample(far, orig_sr=fs_far, target_sr=16000)
        if fs_clean != 16000:
            clean = librosa.resample(clean, orig_sr=fs_clean, target_sr=16000)

        # convert to STFT
        X_mic = wav_to_stft(mic, self.stft_cfg, self.device)  # (F,T,2)
        X_far = wav_to_stft(far, self.stft_cfg, self.device)  # (F,T,2)
        X_clean = wav_to_stft(clean, self.stft_cfg, self.device)  # (F,T,2)

        # segment handling
        # Track valid frame length based on clean target before any right padding
        T_clean_orig = X_clean.shape[1]
        if self.no_crop:
            # Ensure same length across mic/far/clean by right padding per sample
            T_target = max(X_mic.shape[1], X_far.shape[1], X_clean.shape[1])
            X_mic = right_pad_frames(X_mic, T_target)
            X_far = right_pad_frames(X_far, T_target)
            X_clean = right_pad_frames(X_clean, T_target)
            valid_T = T_clean_orig
        else:
            # Random contiguous segment for training
            X_mic = pad_or_crop_frames(X_mic, self.segment_frames)
            X_far = pad_or_crop_frames(X_far, self.segment_frames)
            X_clean = pad_or_crop_frames(X_clean, self.segment_frames)
            # If original shorter than segment_frames, part of segment is padded zeros
            valid_T = min(T_clean_orig, self.segment_frames)

        # to (F,T,2) float32 on device
        # Return valid frame count for clean target to enable masked loss/metrics
        return X_mic, X_far, X_clean, valid_T


def collate_fn(batch):
    # batch items may include valid_T as fourth element
    if len(batch[0]) == 4:
        X_mic_list, X_far_list, X_clean_list, valid_T_list = zip(*batch)
    else:
        X_mic_list, X_far_list, X_clean_list = zip(*batch)
        # Default valid_T equals current T after segmenting
        valid_T_list = [x.shape[1] for x in X_clean_list]
    # stack to (B,F,T,2)
    X_mic = torch.stack(X_mic_list, dim=0)
    X_far = torch.stack(X_far_list, dim=0)
    X_clean = torch.stack(X_clean_list, dim=0)
    valid_Ts = torch.tensor(valid_T_list, dtype=torch.long)
    return X_mic, X_far, X_clean, valid_Ts

def collate_varlen(batch):
    """Pad variable-length (F,T,2) samples in batch to the maximum T across batch.
    Assumes each sample's mic/far/clean already share the same T (handled in dataset when no_crop=True).
    """
    # batch items may include valid_T as fourth element
    if len(batch[0]) == 4:
        X_mic_list, X_far_list, X_clean_list, valid_T_list = zip(*batch)
    else:
        X_mic_list, X_far_list, X_clean_list = zip(*batch)
        # Default valid_T equals current per-sample T
        valid_T_list = [x.shape[1] for x in X_clean_list]
    # Find global max T across all tensors in the batch
    max_T = max(
        max(x.shape[1] for x in X_mic_list),
        max(x.shape[1] for x in X_far_list),
        max(x.shape[1] for x in X_clean_list),
    )

    def pad_to_max(X: torch.Tensor) -> torch.Tensor:
        F, T, C = X.shape
        if T >= max_T:
            return X
        pad = torch.zeros(F, max_T - T, C, device=X.device, dtype=X.dtype)
        return torch.cat([X, pad], dim=1)

    X_mic = torch.stack([pad_to_max(x) for x in X_mic_list], dim=0)
    X_far = torch.stack([pad_to_max(x) for x in X_far_list], dim=0)
    X_clean = torch.stack([pad_to_max(x) for x in X_clean_list], dim=0)
    valid_Ts = torch.tensor(valid_T_list, dtype=torch.long)
    return X_mic, X_far, X_clean, valid_Ts


# =====================
# Loss
# =====================
class SISNRLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss function.
    使用與數據一致的 STFT 參數進行 ISTFT 還原，避免不一致導致失真。
    """
    def __init__(self, stft_cfg: 'STFTConfig' = None, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        # 使用傳入的 STFT 配置或默認配置
        self.stft_cfg = stft_cfg if stft_cfg is not None else STFTConfig()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B,F,T,2) complex spectrograms
        # Convert to time domain for SI-SNR calculation
        pred_time = self.istft(pred)
        target_time = self.istft(target)
        
        # Calculate SI-SNR
        return -self.si_snr(pred_time, target_time)
    
    def istft(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert complex spectrogram to time domain signal using configured STFT params."""
        # spec: (B,F,T,2) -> (B,F,T) complex
        complex_spec = torch.complex(spec[..., 0], spec[..., 1])
        
        n_fft = self.stft_cfg.n_fft
        hop_length = self.stft_cfg.hop_length
        win_length = self.stft_cfg.win_length
        window = self.stft_cfg.window(spec.device)
        
        # ISTFT to get time domain signal
        time_sig = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            length=spec.shape[2] * hop_length  # Approximate length
        )
        return time_sig
    
    def si_snr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate SI-SNR between prediction and target.
        
        Args:
            pred: (B, T) predicted signal
            target: (B, T) target signal
            
        Returns:
            SI-SNR value (higher is better)
        """
        # Zero-mean normalization
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # Projection of pred onto target
        scale = torch.sum(pred * target, dim=-1, keepdim=True) / (torch.sum(target * target, dim=-1, keepdim=True) + self.eps)
        s_target = scale * target
        e_noise = pred - s_target
        
        # Calculate SI-SNR
        si_snr = 10 * torch.log10(
            (torch.sum(s_target * s_target, dim=-1) + self.eps) / 
            (torch.sum(e_noise * e_noise, dim=-1) + self.eps)
        )
        
        # Return mean SI-SNR across batch
        return torch.mean(si_snr)


# Keep the original ComplexSpectralLoss for reference
class ComplexSpectralLoss(nn.Module):
    def __init__(self, mag_weight: float = 0.5):
        super().__init__()
        self.mag_weight = mag_weight
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B,F,T,2)
        complex_l2 = self.mse(pred, target)
        pred_mag = torch.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2 + 1e-12)
        tgt_mag = torch.sqrt(target[..., 0] ** 2 + target[..., 1] ** 2 + 1e-12)
        mag_l1 = self.l1(pred_mag, tgt_mag)
        return complex_l2 + self.mag_weight * mag_l1


# =====================
# Train
# =====================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    stft_cfg = STFTConfig(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)

    # dataset & loader
    ds = AECDataset(
        args.manifest_csv,
        split="train",
        segment_frames=args.segment_frames,
        stft_cfg=stft_cfg,
        device="cpu",
        no_crop=getattr(args, 'train_no_crop', False)
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_varlen if getattr(args, 'train_no_crop', False) else collate_fn,
        drop_last=not getattr(args, 'train_no_crop', False),
        pin_memory=(device.type=='cuda'),
        persistent_workers=True if args.num_workers>0 else False,
    )

    val_dl = None
    if args.use_val:
        val_ds = AECDataset(
            args.manifest_csv,
            split="val",
            segment_frames=args.segment_frames,
            stft_cfg=stft_cfg,
            device="cpu",
            no_crop=getattr(args, 'val_no_crop', False)
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=getattr(args, 'val_batch_size', args.batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_varlen if getattr(args, 'val_no_crop', False) else collate_fn,
            drop_last=False,
            pin_memory=(device.type=='cuda'),
            persistent_workers=True if args.num_workers>0 else False,
        )

    # model
    model = DeepVQE_AEC(align_hidden=2, align_delay=args.align_delay).to(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    model.train()

    # loss, optim
    if args.loss_type == "sisnr":
        criterion = SISNRLoss(stft_cfg=stft_cfg).to(device)
    elif args.loss_type == "complex":
        criterion = ComplexSpectralLoss(mag_weight=0.5).to(device)
    elif args.loss_type == "hybrid":
        criterion = HybridLoss(
            n_fft=args.n_fft,
            hop_len=args.hop_length,
            win_len=args.win_length
        ).to(device)
    else:  # default to sisnr
        criterion = SISNRLoss(stft_cfg=stft_cfg).to(device)
    
    # Advanced optimizer selection
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "radam":
        try:
            from torch.optim import RAdam
            optimizer = RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        except ImportError:
            print("RAdam not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == "sophia":
        optimizer = SophiaG(
            model.parameters(), 
            lr=args.lr, 
            betas=(args.sophia_beta1, args.sophia_beta2),
            rho=args.sophia_rho,
            weight_decay=args.weight_decay,
            k=args.sophia_k
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # Learning rate scheduler (match step-based calling)
    base_scheduler = None
    updates_per_epoch = max(1, math.ceil(len(dl) / max(1, args.accumulate_grad_batches)))
    if args.scheduler == "cosine":
        # Cosine Annealing over total update steps
        total_updates = updates_per_epoch * args.epochs
        base_scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == "step":
        # Decay every ~1/3 of total updates
        step_size_updates = max(1, (updates_per_epoch * args.epochs) // 3)
        base_scheduler = StepLR(optimizer, step_size=step_size_updates, gamma=0.1)
    elif args.scheduler == "exponential":
        base_scheduler = ExponentialLR(optimizer, gamma=0.95)
    
    # Warmup scheduler
    if args.warmup_steps > 0:
        scheduler = WarmupScheduler(optimizer, args.warmup_steps, base_scheduler)
    else:
        scheduler = base_scheduler
    
    # Early stopping
    early_stopping = None
    if args.early_stopping and val_dl is not None:
        early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.cpu) and args.amp)
 
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val = float("inf")
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    val_pesq_scores = []  # backward-compat: store pred PESQ
    val_pesq_pred_scores = []
    val_pesq_clean_scores = []
    val_erle_pred_scores = []
    val_erle_clean_scores = []
    
    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            print(f"Loading checkpoint from {args.resume_checkpoint}")
            # Ensure compatibility with PyTorch 2.6+ where weights_only=True is default
            checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint["model"])
            
            # Load optimizer state if available
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded optimizer state from checkpoint")
            
            # Load scheduler state if available
            if "scheduler" in checkpoint and scheduler is not None:
                if hasattr(scheduler, 'base_scheduler') and scheduler.base_scheduler is not None:
                    if "base_scheduler" in checkpoint:
                        scheduler.base_scheduler.load_state_dict(checkpoint["base_scheduler"])
                        print("Loaded base scheduler state from checkpoint")
                elif "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    print("Loaded scheduler state from checkpoint")
            
            # Load training metrics if available
            if "train_losses" in checkpoint:
                train_losses = checkpoint["train_losses"]
            if "val_losses" in checkpoint:
                val_losses = checkpoint["val_losses"]
            if "val_pesq_scores" in checkpoint:
                val_pesq_scores = checkpoint["val_pesq_scores"]
            
            # Load best validation loss if available
            if "best_val_loss" in checkpoint:
                best_val = checkpoint["best_val_loss"]
                print(f"Loaded best validation loss: {best_val:.4f}")
            
            # Set start epoch
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Resuming training from epoch {start_epoch}")
            
            # Load scaler state if available
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
                print("Loaded scaler state from checkpoint")
                
        else:
            print(f"Warning: Checkpoint file {args.resume_checkpoint} not found. Starting from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        for step, (X_mic, X_far, X_clean, valid_Ts) in enumerate(dl, 1):
            X_mic = X_mic.to(device)
            X_far = X_far.to(device)
            X_clean = X_clean.to(device)
            valid_Ts = valid_Ts.to(device)

            # Gradient accumulation
            if (step - 1) % args.accumulate_grad_batches == 0:
                optimizer.zero_grad(set_to_none=True)
                
            with torch.cuda.amp.autocast(enabled=(not args.cpu) and args.amp):
                Y = model(X_mic, X_far)  # (B,F,T,2)
                # Build frame mask: shape (B,1,T,1) -> broadcast to (B,F,T,2)
                B, F, T, C = Y.shape
                t_idx = torch.arange(T, device=Y.device).unsqueeze(0)  # (1,T)
                valid_mask_bt = (t_idx < valid_Ts.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).float()  # (B,1,T,1)
                # Apply mask to prediction and target for loss/metrics consistency
                Y_masked = Y * valid_mask_bt
                X_clean_masked = X_clean * valid_mask_bt
                
                # Convert to time domain if using HybridLoss
                if args.loss_type == "hybrid":
                    y_pred_time = spec_to_time(Y_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                    y_true_time = spec_to_time(X_clean_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                    loss = criterion(y_pred_time, y_true_time)
                else:
                    loss = criterion(Y_masked, X_clean_masked)
                
                # Scale loss for gradient accumulation
                loss = loss / args.accumulate_grad_batches

            scaler.scale(loss).backward()
            
            # Update weights every accumulate_grad_batches steps
            if step % args.accumulate_grad_batches == 0:
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate scheduler (step-based)
                if scheduler is not None and not isinstance(scheduler.base_scheduler if hasattr(scheduler, 'base_scheduler') else scheduler, ReduceLROnPlateau):
                    scheduler.step()

            running += loss.item() * args.accumulate_grad_batches  # Unscale for logging
            epoch_loss += loss.item() * args.accumulate_grad_batches
            num_batches += 1
            
            if step % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} Step {step}/{len(dl)} | loss={running/args.log_interval:.4f} | lr={current_lr:.2e}")
                running = 0.0
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch} | avg_train_loss={avg_epoch_loss:.4f}")

        # validate
        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                vals = []
                pesq_pred_scores = []
                pesq_clean_scores = []
                erle_pred_scores = []
                erle_clean_scores = []
                for X_mic, X_far, X_clean, valid_Ts in val_dl:
                    X_mic = X_mic.to(device)
                    X_far = X_far.to(device)
                    X_clean = X_clean.to(device)
                    valid_Ts = valid_Ts.to(device)
                    Y = model(X_mic, X_far)
                    # Mask padded frames
                    B, F, T, C = Y.shape
                    t_idx = torch.arange(T, device=Y.device).unsqueeze(0)
                    valid_mask_bt = (t_idx < valid_Ts.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).float()
                    Y_masked = Y * valid_mask_bt
                    X_clean_masked = X_clean * valid_mask_bt
                    
                    # Convert to time domain if using HybridLoss
                    if args.loss_type == "hybrid":
                        y_pred_time = spec_to_time(Y_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        y_true_time = spec_to_time(X_clean_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        y_mic_time = spec_to_time(X_mic, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        vloss = criterion(y_pred_time, y_true_time).item()
                        
                        # Calculate PESQ (pred vs clean) and baseline PESQ (mic vs clean)
                        valid_lengths_time = (valid_Ts * args.hop_length).tolist()
                        pesq_pred = calculate_pesq(y_pred_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                        pesq_clean = calculate_pesq(y_mic_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                        pesq_pred_scores.append(pesq_pred)
                        pesq_clean_scores.append(pesq_clean)
                        
                        # Calculate ERLE (mic vs pred) and baseline (mic vs clean)
                        erle_pred = calculate_erle(y_mic_time, y_pred_time, valid_lengths_time=valid_lengths_time)
                        erle_clean = calculate_erle(y_mic_time, y_true_time, valid_lengths_time=valid_lengths_time)
                        erle_pred_scores.append(erle_pred)
                        erle_clean_scores.append(erle_clean)
                    else:
                        vloss = criterion(Y_masked, X_clean_masked).item()
                        
                        # Convert to time domain for metric calculation
                        y_pred_time = spec_to_time(Y_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        y_true_time = spec_to_time(X_clean_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        y_mic_time = spec_to_time(X_mic, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                        
                        # PESQ metrics
                        valid_lengths_time = (valid_Ts * args.hop_length).tolist()
                        pesq_pred = calculate_pesq(y_pred_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                        pesq_clean = calculate_pesq(y_mic_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                        pesq_pred_scores.append(pesq_pred)
                        pesq_clean_scores.append(pesq_clean)
                        
                        # ERLE metrics
                        erle_pred = calculate_erle(y_mic_time, y_pred_time, valid_lengths_time=valid_lengths_time)
                        erle_clean = calculate_erle(y_mic_time, y_true_time, valid_lengths_time=valid_lengths_time)
                        erle_pred_scores.append(erle_pred)
                        erle_clean_scores.append(erle_clean)
                    
                    vals.append(vloss)
                    
            val_mean = sum(vals) / max(1, len(vals))
            pesq_pred_mean = sum(pesq_pred_scores) / max(1, len(pesq_pred_scores)) if pesq_pred_scores else 0.0
            pesq_clean_mean = sum(pesq_clean_scores) / max(1, len(pesq_clean_scores)) if pesq_clean_scores else 0.0
            erle_pred_mean = sum(erle_pred_scores) / max(1, len(erle_pred_scores)) if erle_pred_scores else 0.0
            erle_clean_mean = sum(erle_clean_scores) / max(1, len(erle_clean_scores)) if erle_clean_scores else 0.0
            
            val_losses.append(val_mean)
            # Backward-compat: store pred PESQ in original list
            val_pesq_scores.append(pesq_pred_mean)
            val_pesq_pred_scores.append(pesq_pred_mean)
            val_pesq_clean_scores.append(pesq_clean_mean)
            val_erle_pred_scores.append(erle_pred_mean)
            val_erle_clean_scores.append(erle_clean_mean)
            print(f"Epoch {epoch} | val_loss={val_mean:.4f} | val_pesq_pred={pesq_pred_mean:.3f} | val_pesq_clean={pesq_clean_mean:.3f} | val_erle_pred={erle_pred_mean:.2f} dB | val_erle_clean={erle_clean_mean:.2f} dB")
            
            # Update learning rate scheduler (validation-based)
            if scheduler is not None and isinstance(scheduler.base_scheduler if hasattr(scheduler, 'base_scheduler') else scheduler, ReduceLROnPlateau):
                if hasattr(scheduler, 'step'):
                    scheduler.step(val_mean)
                else:
                    scheduler.step(val_mean)
            
            # Save best model
            if val_mean < best_val:
                best_val = val_mean
                ckpt_path = os.path.join(args.ckpt_dir, f"deepvqe_aec_best.pt")
                checkpoint_data = {
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "best_val_pesq": pesq_pred_mean,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_pesq_scores": val_pesq_scores,
                    "val_pesq_pred_scores": val_pesq_pred_scores,
                    "val_pesq_clean_scores": val_pesq_clean_scores,
                    "val_erle_pred_scores": val_erle_pred_scores,
                    "val_erle_clean_scores": val_erle_clean_scores,
                    "scaler": scaler.state_dict()
                }
                
                # Save scheduler state
                if scheduler is not None:
                    if hasattr(scheduler, 'base_scheduler') and scheduler.base_scheduler is not None:
                        checkpoint_data["base_scheduler"] = scheduler.base_scheduler.state_dict()
                    else:
                        checkpoint_data["scheduler"] = scheduler.state_dict()
                
                torch.save(checkpoint_data, ckpt_path)
                print(f"Saved best checkpoint to {ckpt_path} (val_loss={best_val:.4f}, val_pesq_pred={pesq_pred_mean:.3f}, val_pesq_clean={pesq_clean_mean:.3f}, val_erle_pred={erle_pred_mean:.2f} dB, val_erle_clean={erle_clean_mean:.2f} dB)")
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_mean, model):
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # save epoch ckpt
        ckpt_path = os.path.join(args.ckpt_dir, f"deepvqe_aec_epoch{epoch}.pt")
        checkpoint_data = {
            "model": model.state_dict(), 
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_pesq_scores": val_pesq_scores,
            "val_pesq_pred_scores": val_pesq_pred_scores,
            "val_pesq_clean_scores": val_pesq_clean_scores,
            "val_erle_pred_scores": val_erle_pred_scores,
            "val_erle_clean_scores": val_erle_clean_scores,
            "scaler": scaler.state_dict()
        }
        
        # Save scheduler state
        if scheduler is not None:
            if hasattr(scheduler, 'base_scheduler') and scheduler.base_scheduler is not None:
                checkpoint_data["base_scheduler"] = scheduler.base_scheduler.state_dict()
            else:
                checkpoint_data["scheduler"] = scheduler.state_dict()
        
        # Save best validation loss if available
        if val_dl is not None:
            checkpoint_data["best_val_loss"] = best_val
        
        torch.save(checkpoint_data, ckpt_path)
        print(f"Saved epoch checkpoint to {ckpt_path}")

    # test after training if requested
    if args.test_after_training:
        test(args, model, stft_cfg, criterion, device)


def test(args, model=None, stft_cfg=None, criterion=None, device=None):
    # Initialize if not provided (e.g., when called directly)
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        stft_cfg = STFTConfig(n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
        model = DeepVQE_AEC(align_hidden=2, align_delay=args.align_delay).to(device)
        # Ensure compatibility with PyTorch 2.6+ where weights_only=True is default
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        
        # Initialize criterion based on loss type
        if args.loss_type == "sisnr":
            criterion = SISNRLoss(stft_cfg=stft_cfg).to(device)
        elif args.loss_type == "complex":
            criterion = ComplexSpectralLoss(mag_weight=0.5).to(device)
        elif args.loss_type == "hybrid":
            criterion = HybridLoss(
                n_fft=args.n_fft,
                hop_len=args.hop_length,
                win_len=args.win_length
            ).to(device)
        else:  # default to sisnr
            criterion = SISNRLoss(stft_cfg=stft_cfg).to(device)

    # test dataset & loader
    test_ds = AECDataset(
        args.manifest_csv,
        split="test",
        segment_frames=args.segment_frames,
        stft_cfg=stft_cfg,
        device="cpu",
        no_crop=getattr(args, 'test_no_crop', False)
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=getattr(args, 'test_batch_size', args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_varlen if getattr(args, 'test_no_crop', False) else collate_fn,
        drop_last=False,
    )

    model.eval()
    with torch.no_grad():
        test_losses = []
        test_pesq_scores = []
        for X_mic, X_far, X_clean, valid_Ts in test_dl:
            X_mic = X_mic.to(device)
            X_far = X_far.to(device)
            X_clean = X_clean.to(device)
            valid_Ts = valid_Ts.to(device)
            Y = model(X_mic, X_far)
            # Mask padded frames
            B, F, T, C = Y.shape
            t_idx = torch.arange(T, device=Y.device).unsqueeze(0)
            valid_mask_bt = (t_idx < valid_Ts.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).float()
            Y_masked = Y * valid_mask_bt
            X_clean_masked = X_clean * valid_mask_bt
            
            # Handle different loss types
            if args.loss_type == "hybrid":
                y_pred_time = spec_to_time(Y_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                y_true_time = spec_to_time(X_clean_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                test_loss = criterion(y_pred_time, y_true_time).item()
                
                # Calculate PESQ for time domain signals
                valid_lengths_time = (valid_Ts * args.hop_length).tolist()
                pesq_score = calculate_pesq(y_pred_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                test_pesq_scores.append(pesq_score)
            else:
                test_loss = criterion(Y_masked, X_clean_masked).item()
                
                # Convert to time domain for PESQ calculation
                y_pred_time = spec_to_time(Y_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                y_true_time = spec_to_time(X_clean_masked, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length)
                valid_lengths_time = (valid_Ts * args.hop_length).tolist()
                pesq_score = calculate_pesq(y_pred_time, y_true_time, fs=16000, valid_lengths_time=valid_lengths_time)
                test_pesq_scores.append(pesq_score)
            
            test_losses.append(test_loss)
            
    test_mean = sum(test_losses) / max(1, len(test_losses))
    test_pesq_mean = sum(test_pesq_scores) / max(1, len(test_pesq_scores)) if test_pesq_scores else 0.0
    print(f"Test loss: {test_mean:.4f} | Test PESQ: {test_pesq_mean:.3f}")
    return test_mean, test_pesq_mean


def build_argparser():
    p = argparse.ArgumentParser(description="Train DeepVQE-AEC with AlignBlock and Advanced Optimization")
    
    # Data and model arguments
    p.add_argument("--manifest_csv", type=str, default="train.csv", help="CSV with columns mix_filepath,farnerd_filepath,target_filepath,split")
    p.add_argument("--use_val", action=argparse.BooleanOptionalAction, default=True, help="Use validation set from the manifest CSV")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--segment_frames", type=int, default=63, help="number of STFT frames per sample segment")
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--win_length", type=int, default=512)
    p.add_argument("--align_delay", type=int, default=200)
    p.add_argument("--loss_type", type=str, default="hybrid", choices=["sisnr", "complex", "hybrid"], help="Loss function type: sisnr, complex, or hybrid")
    
    # Training arguments
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--val_batch_size", type=int, default=8, help="Batch size for validation (use smaller value to avoid OOM with val_no_crop)")
    p.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing (use smaller value to avoid OOM with test_no_crop)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--accumulate_grad_batches", type=int, default=1, help="Number of batches to accumulate gradients")
    p.add_argument("--clip_grad", type=float, default=5.0)
    p.add_argument("--log_interval", type=int, default=50)
    
    # Optimizer arguments
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "radam", "sgd", "sophia"], help="Optimizer type")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for regularization")
    
    # Sophia optimizer specific arguments
    p.add_argument("--sophia_beta1", type=float, default=0.965, help="Sophia beta1 parameter")
    p.add_argument("--sophia_beta2", type=float, default=0.99, help="Sophia beta2 parameter")
    p.add_argument("--sophia_rho", type=float, default=0.04, help="Sophia rho parameter for Hessian scaling")
    p.add_argument("--sophia_k", type=int, default=10, help="Sophia k parameter for Hessian update interval")
    
    # Learning rate scheduler arguments
    p.add_argument("--scheduler", type=str, default="plateau", choices=["cosine", "plateau", "step", "exponential", "none"], help="Learning rate scheduler")
    p.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for learning rate")
    
    # Early stopping arguments
    p.add_argument("--early_stopping", action="store_true", default=False, help="Enable early stopping")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    p.add_argument("--min_delta", type=float, default=1e-4, help="Minimum change to qualify as an improvement")
    
    # System arguments
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    
    # Testing arguments
    p.add_argument("--test_after_training", action="store_true", default=True, help="Run test after training")
    p.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint for testing")
    p.add_argument("--test_only", action="store_true", help="Only run test, no training")
    # Validation/Test no-crop options
    p.add_argument("--val_no_crop", action=argparse.BooleanOptionalAction, default=True, help="Use full-length segments with right padding for validation")
    p.add_argument("--test_no_crop", action="store_true", help="Use full-length segments with right padding for testing")
    # Training no-crop option
    p.add_argument("--train_no_crop", action=argparse.BooleanOptionalAction, default=True, help="Use full-length segments with right padding for training")
    
    # Resume training arguments
    p.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint to resume training from")
    
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.test_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint must be provided when using --test_only")
        test(args)
    else:
        train(args)
        print()
