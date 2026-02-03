"""Model surgery for extracting CoreML-compatible neural network from HTDemucs.

This module extracts the neural network portion of HTDemucs that processes
both spectrograms and raw audio, enabling the full hybrid model with both
frequency and time branches.

HTDemucs is a hybrid model where:
    final_output = time_branch + istft(frequency_branch)

This module provides FullHybridHTDemucs which accepts both inputs and returns
both outputs, allowing Swift to handle STFT/iSTFT externally while CoreML
handles both neural network branches.

CoreML conversion is possible because complex number operations are handled
externally in Swift using the Accelerate framework.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class FullHybridHTDemucs(nn.Module):
    """Full hybrid HTDemucs with both frequency and time branches.

    This module wraps HTDemucs to process both spectrograms and raw audio,
    running the complete hybrid architecture including:
    1. Frequency encoder on spectrogram
    2. Time encoder on raw audio
    3. Cross-domain transformer on both branches
    4. Frequency decoder returning separated spectrograms
    5. Time decoder returning separated audio

    The final separation is: output = time_output + istft(freq_output)
    This addition is performed in Swift after CoreML inference.

    Attributes:
        htdemucs: The original HTDemucs model.
        num_sources: Number of sources (6 for htdemucs_6s).
        audio_channels: Number of audio channels (2 for stereo).
    """

    def __init__(
        self,
        htdemucs: nn.Module,
        audio_length: Optional[int] = None,
        freq_bins: Optional[int] = None,
        time_frames: Optional[int] = None,
    ):
        """Initialize FullHybridHTDemucs by wrapping an HTDemucs model.

        Args:
            htdemucs: An HTDemucs model instance (typically loaded via
                demucs.pretrained.get_model).
            audio_length: Fixed audio length for CoreML compatibility. If None,
                uses the model's training segment length (343980 samples for 7.8s).
            freq_bins: Fixed frequency bins in output spectrogram. If None,
                defaults to 2048 (htdemucs default).
            time_frames: Fixed time frames in output spectrogram. If None,
                computed from audio_length using STFT parameters.
        """
        super().__init__()
        self.htdemucs = htdemucs
        # Store constants for CoreML compatibility (avoid dynamic shape ops)
        self.num_sources = len(htdemucs.sources)  # 6
        self.audio_channels = htdemucs.audio_channels  # 2
        self.cac_channels = self.audio_channels * 2  # 4 (real+imag for stereo)

        # Fixed audio length for CoreML (avoids dynamic shape extraction)
        if audio_length is None:
            audio_length = int(htdemucs.segment * htdemucs.samplerate)
        self.audio_length = audio_length

        # Fixed frequency output dimensions (encoder/decoder preserve dimensions)
        if freq_bins is None:
            freq_bins = 2048  # htdemucs uses nfft=4096, so freq_bins = nfft//2
        self.freq_bins = freq_bins

        # Fixed time frames in spectrogram output
        # NOTE: time_frames must match actual STFT output dimensions.
        # For htdemucs with training_length=343980, the actual is 336.
        # Best practice: compute from a sample STFT or pass explicit value.
        if time_frames is None:
            # Default to 336 which matches htdemucs training segment
            time_frames = 336
        self.time_frames = time_frames

    def forward(
        self, spectrogram: torch.Tensor, raw_audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process spectrogram and raw audio through both branches.

        Args:
            spectrogram: Complex-as-Channels spectrogram with shape:
                (batch, channels=4, freq_bins=2049, time_frames=431)
                where channels are [real_L, imag_L, real_R, imag_R]
            raw_audio: Raw stereo audio waveform with shape:
                (batch, channels=2, samples=441000)

        Returns:
            Tuple of (freq_output, time_output):
            - freq_output: Separated spectrograms with shape
                (batch, sources=6, channels=4, freq_bins=2048, time_frames=431)
            - time_output: Separated audio with shape
                (batch, sources=6, channels=2, samples=441000)
        """
        B, C, Fq, T = spectrogram.shape
        # Use fixed length constant for CoreML compatibility
        length = self.audio_length

        # === Normalize frequency branch (spectrogram) ===
        mean = spectrogram.mean(dim=(1, 2, 3), keepdim=True)
        std = spectrogram.std(dim=(1, 2, 3), keepdim=True)
        x = (spectrogram - mean) / (1e-5 + std)

        # === Normalize time branch (raw audio) ===
        meant = raw_audio.mean(dim=(1, 2), keepdim=True)
        stdt = raw_audio.std(dim=(1, 2), keepdim=True)
        xt = (raw_audio - meant) / (1e-5 + stdt)

        # === Encoder: process both branches ===
        saved = []  # Skip connections for frequency branch
        saved_t = []  # Skip connections for time branch
        lengths = []  # Saved lengths for frequency decoder
        lengths_t = []  # Saved lengths for time decoder

        for idx, encode in enumerate(self.htdemucs.encoder):
            lengths.append(x.shape[-1])
            inject = None

            if idx < len(self.htdemucs.tencoder):
                # Process time branch
                lengths_t.append(xt.shape[-1])
                tenc = self.htdemucs.tencoder[idx]
                xt = tenc(xt)

                if not tenc.empty:
                    # Save for skip connection
                    saved_t.append(xt)
                else:
                    # First conv merges time into freq branch
                    inject = xt

            # Process frequency branch with injection from time branch
            x = encode(x, inject)

            # Add frequency embedding after first encoder layer
            if idx == 0 and self.htdemucs.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.htdemucs.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.htdemucs.freq_emb_scale * emb

            saved.append(x)

        # === Cross-transformer: exchange information between branches ===
        if self.htdemucs.crosstransformer:
            if self.htdemucs.bottom_channels:
                from einops import rearrange

                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_upsampler_t(xt)

            x, xt = self.htdemucs.crosstransformer(x, xt)

            if self.htdemucs.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.htdemucs.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.htdemucs.channel_downsampler_t(xt)

        # === Decoder: process both branches ===
        for idx, decode in enumerate(self.htdemucs.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            # Time decoder starts later (offset by depth difference)
            offset = self.htdemucs.depth - len(self.htdemucs.tdecoder)
            if idx >= offset:
                tdec = self.htdemucs.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)

                if tdec.empty:
                    # Use pre-transconv output from freq branch
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        # Verify all skip connections were used
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        # === Reshape and denormalize outputs ===
        # Use stored constants for CoreML compatibility
        S = self.num_sources  # 6
        C_per_source = self.cac_channels  # 4

        # Frequency branch output: (B, S*4, freq, time) -> (B, S, 4, freq, time)
        # Use fixed dimensions for CoreML compatibility (no dynamic size extraction)
        freq_output = x.view(1, S, C_per_source, self.freq_bins, self.time_frames)

        # Denormalize frequency output
        freq_output = freq_output * std[:, None] + mean[:, None]

        # Time branch output: (B, S*2, length) -> (B, S, 2, length)
        time_output = xt.view(1, S, self.audio_channels, length)
        time_output = time_output * stdt[:, None] + meant[:, None]

        return freq_output, time_output


# Keep InnerHTDemucs for backwards compatibility
class InnerHTDemucs(nn.Module):
    """HTDemucs neural network without STFT/iSTFT operations (frequency branch only).

    DEPRECATED: Use FullHybridHTDemucs for proper hybrid model support.

    This module extracts only the frequency branch from HTDemucs, passing zeros
    for the time branch. This results in significant quality loss since HTDemucs
    relies on both branches.
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


def extract_full_hybrid_model(
    htdemucs: nn.Module,
    audio_length: Optional[int] = None,
    freq_bins: Optional[int] = None,
    time_frames: Optional[int] = None,
) -> FullHybridHTDemucs:
    """Extract the full hybrid neural network from an HTDemucs model.

    Wraps an HTDemucs model to create a version that accepts both spectrograms
    and raw audio, processing both frequency and time branches. This is the
    recommended approach for accurate source separation.

    The wrapper removes STFT/iSTFT operations (handled in Swift) but preserves
    both neural network branches for proper hybrid processing.

    Args:
        htdemucs: An HTDemucs model instance. Can be obtained via:
            from demucs.pretrained import get_model
            model = get_model('htdemucs_6s')
            hybrid = extract_full_hybrid_model(model.models[0])
        audio_length: Fixed audio length for CoreML compatibility. If None,
            uses the model's training segment length (343980 samples for 7.8s).
        freq_bins: Fixed frequency bins in output spectrogram. If None,
            defaults to 2048.
        time_frames: Fixed time frames in output spectrogram. If None,
            computed from audio_length.

    Returns:
        FullHybridHTDemucs: A wrapped model that processes both branches.

    Example:
        >>> from demucs.pretrained import get_model
        >>> model = get_model('htdemucs_6s')
        >>> hybrid_model = extract_full_hybrid_model(model.models[0])
        >>> hybrid_model.eval()
        >>> # Inputs:
        >>> #   spectrogram: (batch=1, channels=4, freq=2048, time=336)
        >>> #   raw_audio: (batch=1, channels=2, samples=343980)
        >>> # Outputs:
        >>> #   freq_output: (batch=1, sources=6, channels=4, freq=2048, time=336)
        >>> #   time_output: (batch=1, sources=6, channels=2, samples=343980)
    """
    return FullHybridHTDemucs(
        htdemucs,
        audio_length=audio_length,
        freq_bins=freq_bins,
        time_frames=time_frames,
    )


def extract_inner_model(htdemucs: nn.Module) -> InnerHTDemucs:
    """Extract the inner neural network from an HTDemucs model (frequency branch only).

    DEPRECATED: Use extract_full_hybrid_model() for proper hybrid model support.
    This function only extracts the frequency branch, resulting in significant
    quality loss.

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


def capture_stft_output(htdemucs_model: nn.Module, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture intermediate STFT output from HTDemucs forward pass using hooks.

    This function intercepts the complex spectrogram output from the HTDemucs STFT
    operation (_spec method) and separates it into real and imaginary components.

    The STFT is computed with:
    - FFT size (nfft): 4096
    - Hop length: 1024
    - Number of frequency bins: 2049 (nfft // 2 + 1)

    Args:
        htdemucs_model: The HTDemucs model (can be BagOfModels wrapper or HTDemucs directly).
            If BagOfModels, extracts the first model internally.
        audio: Input audio tensor with shape (batch, channels, samples).
            Expected to be stereo (channels=2) for HTDemucs.

    Returns:
        Tuple of (real_part, imag_part) where:
        - real_part: Tensor of shape (batch, channels, freq_bins=2049, time_frames)
        - imag_part: Tensor of shape (batch, channels, freq_bins=2049, time_frames)

    Example:
        >>> from demucs.pretrained import get_model
        >>> model = get_model('htdemucs_6s')
        >>> audio = torch.randn(1, 2, 44100)  # 1 second at 44.1kHz
        >>> real, imag = capture_stft_output(model, audio)
        >>> print(real.shape)  # torch.Size([1, 2, 2049, time_frames])
    """
    # Handle both BagOfModels wrapper and raw HTDemucs
    if hasattr(htdemucs_model, 'models'):
        # This is a BagOfModels wrapper, get the first model
        actual_model = htdemucs_model.models[0]
    else:
        # This is already an HTDemucs model
        actual_model = htdemucs_model

    # Container to capture the spectrogram
    captured_spectrogram: Optional[torch.Tensor] = None

    def capture_hook(module, input, output):
        """Hook to capture the STFT output."""
        nonlocal captured_spectrogram
        # output from _spec is the complex spectrogram with shape (batch, channels, freq, time)
        captured_spectrogram = output

    # Register hook on the _spec method
    # We need to hook the method itself, not the module
    # Instead, we'll hook the torch.stft operation indirectly by examining the forward pass
    # Actually, we need to hook when _spec returns its result
    # Let's use a simpler approach: monkey-patch the _spec method temporarily

    original_spec = actual_model._spec

    def hooked_spec(x):
        """Wrapper around _spec that captures its output."""
        nonlocal captured_spectrogram
        z = original_spec(x)
        captured_spectrogram = z
        return z

    # Temporarily replace _spec with our hooked version
    actual_model._spec = hooked_spec

    try:
        with torch.no_grad():
            # Run the forward pass to trigger the STFT
            _ = actual_model(audio)
    finally:
        # Restore the original _spec method
        actual_model._spec = original_spec

    # Extract real and imaginary parts from the complex spectrogram
    if captured_spectrogram is None:
        raise RuntimeError("Failed to capture STFT output")

    # Convert complex tensor to real and imaginary components
    # captured_spectrogram shape: (batch, channels, freq_bins, time_frames)
    real_part = captured_spectrogram.real
    imag_part = captured_spectrogram.imag

    return real_part, imag_part
