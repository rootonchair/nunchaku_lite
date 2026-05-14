"""Quantized linear modules backed by Nunchaku Lite native kernels."""

import torch
from torch import nn

from ..ops.gemm import svdq_gemm_w4a4_cuda
from ..ops.gemv import awq_gemv_w4a16_cuda
from ..ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda


class SVDQW4A4Linear(nn.Module):
    """SVDQ W4A4 linear projection with low-rank correction parameters.

    The module owns the parameter buffers expected by Nunchaku SVDQ
    checkpoints. Parameters are allocated empty and are populated later through
    ``load_state_dict``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        """Allocate SVDQ parameter buffers for a quantized linear projection.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            rank: Low-rank correction rank.
            bias: Whether to allocate a bias parameter.
            precision: Native weight precision, either ``"int4"`` or
                ``"nvfp4"``.
            act_unsigned: Whether the activation quantization path should use
                unsigned activations.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Device for parameter allocation.

        Raises:
            ValueError: If ``precision`` is unsupported.

        Returns:
            None.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.torch_dtype = torch_dtype

        if precision == "nvfp4":
            self.group_size = 16
        elif precision == "int4":
            self.group_size = 64
        else:
            raise ValueError(f"Invalid precision: {precision}")

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                in_features // self.group_size,
                out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
            self.wtscale = 1.0
        else:
            self.wtscale = None
            self.wcscales = None

        self.act_unsigned = act_unsigned

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        """Create an empty SVDQ module with metadata copied from ``linear``.

        Args:
            linear: Source dense linear layer.
            **kwargs: Constructor overrides such as ``rank``, ``precision``,
                ``torch_dtype``, ``device``, or ``in_features``.

        Returns:
            New :class:`SVDQW4A4Linear` with matching output shape and bias
            presence.
        """

        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        device = kwargs.pop("device", linear.weight.device)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=device,
            **kwargs,
        )

    def quantize(self, x: torch.Tensor, pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize activations and compute the low-rank activation side output.

        Args:
            x: Flattened activation tensor with shape ``(tokens, channels)``.
            pad_size: Token padding multiple required by the native quantizer.

        Returns:
            Tuple of quantized activations, activation scales, and low-rank
            activation output.
        """

        return svdq_quantize_w4a4_act_fuse_lora_cuda(
            x,
            lora_down=self.proj_down,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
            pad_size=pad_size,
        )

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run native W4A4 GEMM on already-quantized activations.

        Args:
            quantized_x: Quantized activation tensor returned by
                :meth:`quantize`.
            ascales: Activation scales returned by :meth:`quantize`.
            lora_act: Low-rank activation tensor returned by :meth:`quantize`.
            output: Optional preallocated output tensor.

        Returns:
            Projection output tensor.
        """

        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )
        svdq_gemm_w4a4_cuda(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the quantized projection to a batched sequence tensor.

        Args:
            x: Input tensor with shape ``(batch, sequence, channels)``.
            output: Optional flattened preallocated output buffer.

        Returns:
            Output tensor with shape ``(batch, sequence, out_features)``.
        """

        batch_size, seq_len, channels = x.shape
        x = x.reshape(batch_size * seq_len, channels)
        if output is None:
            output = torch.empty(batch_size * seq_len, self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        return output.reshape(batch_size, seq_len, -1)

    def __repr__(self):
        """Return a compact representation with quantization metadata.

        Args:
            None.

        Returns:
            Debug string containing feature sizes, rank, precision, and
            activation signedness.
        """

        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


class AWQW4A16Linear(nn.Module):
    """AWQ W4A16 linear projection used by selected Flux adapter paths.

    This module stores packed AWQ checkpoint buffers and dispatches to the
    native GEMV kernel at runtime.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        """Create an empty AWQ linear module with packed weight buffers.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            bias: Whether to allocate a bias parameter.
            group_size: AWQ scale/zero group size.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Device for parameter allocation.

        Returns:
            None.
        """

        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.qweight = nn.Parameter(
            torch.empty(out_features // 4, in_features // 2, dtype=torch.int32, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wzeros = nn.Parameter(
            torch.empty(in_features // group_size, out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 64,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        """Create an empty AWQ module with metadata copied from ``linear``.

        Args:
            linear: Source dense linear layer.
            group_size: AWQ group size.
            torch_dtype: Runtime dtype for floating-point buffers.
            device: Optional allocation device. Defaults to the source
                linear's device.
            **kwargs: Ignored compatibility kwargs accepted by adapter helpers.

        Returns:
            New :class:`AWQW4A16Linear`.
        """

        if device is None:
            device = linear.weight.device
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            torch_dtype=torch_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the AWQ projection using the native GEMV kernel.

        Args:
            x: Input activation tensor whose last dimension is
                ``in_features``.

        Returns:
            Output tensor with last dimension ``out_features``.
        """

        output = awq_gemv_w4a16_cuda(
            in_feats=x,
            kernel=self.qweight,
            scaling_factors=self.wscales,
            zeros=self.wzeros,
            m=x.shape[0],
            n=self.out_features,
            k=self.in_features,
            group_size=self.group_size,
        )
        if self.bias is not None:
            output.add_(self.bias.view([1] * (output.ndim - 1) + [-1]))
        lora_down = getattr(self, "_nunchaku_lite_lora_down", None)
        lora_up = getattr(self, "_nunchaku_lite_lora_up", None)
        if lora_down is not None and lora_up is not None and lora_down.shape[1] > 0:
            if lora_down.device != x.device:
                lora_down = lora_down.to(x.device)
                self._nunchaku_lite_lora_down = lora_down
            if lora_up.device != x.device:
                lora_up = lora_up.to(x.device)
                self._nunchaku_lite_lora_up = lora_up
            lora = torch.matmul(x.to(lora_down.dtype), lora_down)
            lora = torch.matmul(lora, lora_up.transpose(0, 1))
            output.add_(lora.to(output.dtype))
        return output

    def __repr__(self):
        """Return a compact representation with AWQ metadata.

        Args:
            None.

        Returns:
            Debug string containing feature sizes and group size.
        """

        return (
            f"AWQW4A16Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size})"
        )
