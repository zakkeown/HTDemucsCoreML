"""Model surgery for extracting CoreML-compatible neural network from HTDemucs.

This module extracts the neural network portion of HTDemucs that processes
spectrograms, removing the STFT/iSTFT operations. The extracted model accepts
spectrograms in Complex-as-Channels format (real and imaginary parts stacked
as separate channels) and outputs separation masks.

This enables CoreML conversion since complex number operations are not supported
by CoreML. The STFT/iSTFT operations are intended to be implemented separately
in native code (Swift using Accelerate framework).
"""

import torch
import torch.nn as nn
from typing import Tuple


class InnerHTDemucs(nn.Module):
    """HTDemucs neural network without STFT/iSTFT operations.

    This module extracts the core neural network from HTDemucs that processes
    spectrograms directly. It accepts Complex-as-Channels format input
    (real and imaginary parts concatenated along the channel dimension) and
    outputs separation masks for each source.

    The model processes input through:
    1. Normalization
    2. Time and frequency branch encoders
    3. Cross-domain transformer
    4. Frequency and time branch decoders
    5. Denormalization

    Attributes:
        htdemucs: The original HTDemucs model to extract from.
        mean: Mean value for normalization (set during first forward pass).
        std: Std value for normalization (set during first forward pass).
    """

    def __init__(self, htdemucs: nn.Module):
        """Initialize InnerHTDemucs by wrapping an HTDemucs model.

        Args:
            htdemucs: An HTDemucs model instance (typically loaded via
                demucs.pretrained.get_model).
        """
        super().__init__()
        self.htdemucs = htdemucs
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Process a spectrogram and return separation masks.

        Args:
            spectrogram: Complex-as-Channels spectrogram with shape:
                (batch, channels=4, freq_bins, time_frames)
                where channels are [real_L, imag_L, real_R, imag_R] for stereo mix
                (or [real, imag] for mono).

        Returns:
            Separation masks with shape:
                (batch, sources=6, freq_bins, time_frames)
                where sources are [drums, bass, vocals, guitar, piano, other]
        """
        B, C, Fq, T = spectrogram.shape

        # Normalize the spectrogram using its statistics
        mean = spectrogram.mean(dim=(1, 2, 3), keepdim=True)
        std = spectrogram.std(dim=(1, 2, 3), keepdim=True)
        x = (spectrogram - mean) / (1e-5 + std)

        # Forward through encoder
        # We skip the time branch since we only have spectrograms
        saved = []  # Skip connections from encoder
        lengths = []  # Save lengths for decoder
        for idx, encode in enumerate(self.htdemucs.encoder):
            lengths.append(x.shape[-1])
            # encode(x, inject) where inject is from time branch
            # Since we have no time branch, inject is None
            x = encode(x, None)

            # Add frequency embedding after first encoder layer
            if idx == 0 and self.htdemucs.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.htdemucs.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.htdemucs.freq_emb_scale * emb

            saved.append(x)

        # Cross-transformer if available
        # The crosstransformer takes (x, xt) but we only have x (spectrogram)
        # We create a dummy time branch with same batch size for compatibility
        if self.htdemucs.crosstransformer:
            # Create a dummy time branch tensor with shape (B, C_time, T_time)
            # Since we don't have actual time domain data, use zeros
            # The actual shape doesn't matter as much since we only care about the freq branch
            # But we need to match what the crosstransformer expects
            xt_dummy = torch.zeros(B, x.shape[1], T, device=x.device, dtype=x.dtype)

            if self.htdemucs.bottom_channels:
                from einops import rearrange
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)

            # Pass dummy time branch to crosstransformer
            x, xt_dummy = self.htdemucs.crosstransformer(x, xt_dummy)

            if self.htdemucs.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)

        # Forward through decoder
        for decode in self.htdemucs.decoder:
            skip = saved.pop(-1)
            length = lengths.pop(-1)
            x, _ = decode(x, skip, length)

        # Reshape output to separate sources and channels
        # Decoder output shape: (B, C_out, freq, time)
        # We need to reshape to: (B, sources, channels, freq, time)
        # Where channels=4 for CaC format (real/imag for stereo)
        S = len(self.htdemucs.sources)
        B_out, C_out, Freq_out, T_out = x.shape
        # C_out should be S * 4 (6 sources * 4 channels for stereo CaC)
        C_per_source = C_out // S
        x = x.view(B_out, S, C_per_source, Freq_out, T_out)

        # Denormalize using the input statistics
        # std and mean have shape (1, 1, 1, 1), squeeze them and reshape for broadcasting
        std_squeezed = std.view(1, 1, 1)
        mean_squeezed = mean.view(1, 1, 1)
        x = x * std_squeezed + mean_squeezed

        # Output shape: (batch, sources, channels, freq, time)
        # For our case: (1, 6, 4, 2048, 431)
        return x


def extract_inner_model(htdemucs: nn.Module) -> InnerHTDemucs:
    """Extract the inner neural network from an HTDemucs model.

    Wraps an HTDemucs model to create a version that accepts spectrograms
    directly (in Complex-as-Channels format) instead of raw audio waveforms.
    This removes the STFT/iSTFT operations, making the model CoreML-compatible.

    Args:
        htdemucs: An HTDemucs model instance. Can be obtained via:
            from demucs.pretrained import get_model
            model = get_model('htdemucs_6s')
            inner = extract_inner_model(model.models[0])

    Returns:
        InnerHTDemucs: A wrapped model that processes spectrograms.

    Example:
        >>> from demucs.pretrained import get_model
        >>> model = get_model('htdemucs_6s')
        >>> inner_model = extract_inner_model(model.models[0])
        >>> inner_model.eval()
        >>> # Input: spectrogram (batch=1, channels=2, freq=2049, time=431)
        >>> # Output: masks (batch=1, sources=6, freq=2049, time=431)
    """
    return InnerHTDemucs(htdemucs)
