"""Transformer encoder and decoder for the de novo sequencing task."""

import math
from collections.abc import Callable
from typing import Optional

import torch
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)


class FourierFeatureEncoder(torch.nn.Module):
    """Fourier Feature Embeddings

    Parameters
    ----------
    d_model : int
        Size of the third dimension of the output tensor.
    m_max : Optional[float], default=10000
        Maximum fourier feature frequency. Should be in the range (1, inf).
    m_min : Optional[float], default=10**-4
        Minimum fourier feature frequency. Should be in the range (0, 1)
    """

    def __init__(
        self,
        d_out: int,
        m_max: Optional[float] = 1000,
        m_min: Optional[float] = 10**-4,
    ) -> None:
        """Initialize the MassEncoder."""
        super().__init__()
        max_component = 1.0 / torch.arange(m_max, 0, -1).float()
        min_component = 1.0 / (
            m_min * torch.arange(round(1 / m_min), 0, -1).float()
        )
        frequencies = (
            2.0 * torch.pi * torch.cat((max_component, min_component))
        )
        self.register_buffer("frequencies", frequencies.view(1, 1, -1))
        self.dim_project = torch.nn.Linear(2 * len(frequencies), d_out)

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
        trig_inputs = X.unsqueeze(-1) * self.frequencies
        fourier_features = torch.cat(
            (torch.sin(trig_inputs), torch.cos(trig_inputs)), dim=-1
        )
        return self.dim_project(fourier_features)


class FourierPeakEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        m_max: Optional[float] = 1000,
        m_min: Optional[float] = 0.0002,
    ) -> None:
        super().__init__()
        fourier_encoder_dim = math.ceil(d_model / 2)
        mz_int_dim = d_model - fourier_encoder_dim
        self.fourier_mz_encoder = FourierFeatureEncoder(
            fourier_encoder_dim, m_max=m_max, m_min=m_min
        )
        self.mz_int_proj = torch.nn.Linear(2, mz_int_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the mass spectra.

        """
        return torch.cat(
            (self.fourier_mz_encoder(X[:, :, 0]), self.mz_int_proj(X)),
            dim=-1,
        )


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
