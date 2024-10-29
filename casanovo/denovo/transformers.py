"""Transformer encoder and decoder for the de novo sequencing task."""

from collections.abc import Callable
from typing import Optional

import einops
import torch
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)


class FourierFloatEncoder(torch.nn.Module):
    """Fourier Float Value Embeddings

    Parameters
    ----------
    d_model : int
        The dimensionality of the output embedding.

    start_exp : Optional[int], optional, default=1
        The starting exponent used to define the range of Fourier frequencies.

    weave : bool, optional, default=False
        If set to True, the encoder will interleave sine and cosine components
        in the embeddings tensor, else concatenate sine and cosine components.
    """

    def __init__(
        self,
        d_model: int,
        start_exp: Optional[int] = 1,
        weave: Optional[bool] = False,
    ) -> None:
        """Initialize the MassEncoder."""
        super().__init__()

        # Get dimensions for equations:
        d_sin = math.ceil(d_model / 2)
        d_cos = d_model - d_sin

        self.wave_lengths = (
            2
            ** torch.arange(
                start_exp, start_exp - max(d_sin, d_cos), -1
            ).float()
        )

        self.weave = weave
        self.register_buffer("sin_term", self.wave_lengths[:d_sin].clone())
        self.register_buffer("cos_term", self.wave_lengths[:d_cos].clone())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_float)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_float, d_model)
            The encoded features for the floating point numbers.

        """
        sin_features = torch.sin(X[:, :, None] * self.sin_term)
        cos_features = torch.cos(X[:, :, None] * self.cos_term)

        if self.weave:
            result = torch.empty(
                (*X.shape, len(self.sin_term) + len(self.cos_term))
            )
            result[:, :, 0::2] = sin_features
            result[:, :, 1::2] = cos_features
            return result
        else:
            return torch.cat([sin_features, cos_features], axis=-1)


class FourierPeakEncoder(PeakEncoder):
    def __init__(
        self,
        d_model: int,
        start_exp: Optional[int] = 1,
        weave: Optional[bool] = False,
    ) -> None:
        super().__init__(d_model)
        self.mz_encoder = FourierFloatEncoder(d_model, start_exp, weave)
        self.int_encoder = FourierFloatEncoder(d_model, start_exp, weave)


class FourierPositionalEncoder(FourierFloatEncoder):
    def __init__(
        self,
        d_model: int,
        start_exp: Optional[int] = 1,
        weave: Optional[bool] = False,
    ) -> None:
        super().__init__(d_model, start_exp, weave)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.

        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in * self.sin_term)
        cos_pos = torch.cos(cos_in * self.cos_term)

        if self.weave:
            encoded = torch.empty(X.shape)
            encoded[:, :, 0::2] = sin_pos
            encoded[:, :, 1::2] = cos_pos
        else:
            encoded = torch.cat([sin_pos, cos_pos], axis=2)

        return encoded + X


class PeptideDecoder(AnalyteTransformerDecoder):
    """A transformer decoder for peptide sequences

    Parameters
    ----------
    n_tokens : int
        The number of tokens used to tokenize peptide sequences.
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    pos_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        max_charge: int = 10,
    ) -> None:
        """Initialize a PeptideDecoder."""

        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
            padding_int=padding_int,
        )

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mass_encoder = FloatEncoder(d_model)

        # override final layer:
        # +1 in comparison to version in depthcharge to second dimension
        # This includes padding (=0) as a possible class
        # and avoids problems during beam search decoding
        self.final = torch.nn.Linear(
            d_model,
            self.token_encoder.num_embeddings,
        )

    def global_token_hook(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include precursor information.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor
            Precursor information.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The global token representations.

        """
        masses = self.mass_encoder(precursors[:, None, 0]).squeeze(1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges
        return precursors


class SpectrumEncoder(SpectrumTransformerEncoder):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``d_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__(
            d_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder
        )

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Override global_token_hook to include
        lantent_spectrum parameter

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.

        """
        return self.latent_spectrum.squeeze(0).expand(mz_array.shape[0], -1)
