"""Runtime manifest parsing for Nunchaku Lite checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA = "nunchaku_lite.runtime_manifest"
SUPPORTED_VERSION = 1
SUPPORTED_FORMAT_VERSION = 1
SUPPORTED_OPS = {"svdq_w4a4", "awq_w4a16", "adanorm_awq_w4a16"}
SUPPORTED_PRECISIONS = {"int4", "fp4"}
SUPPORTED_KINDS = {"linear"}


@dataclass(frozen=True)
class RuntimeManifestTarget:
    """One runtime target declared by a Nunchaku Lite manifest."""

    name: str | None
    checkpoint_prefix: str
    source_modules: tuple[str, ...]
    roles: tuple[str, ...]
    kind: str
    nunchaku_op: str
    precision: str
    group_size: int
    rank: int
    has_bias: bool
    op_options: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)

    @property
    def runtime_precision(self) -> str:
        """Return the native precision name used by runtime modules."""

        return "nvfp4" if self.precision == "fp4" else self.precision


@dataclass(frozen=True)
class RuntimeManifest:
    """Validated v1 Nunchaku Lite runtime manifest."""

    schema: str
    version: int
    component: str
    nunchaku_format_version: int
    producer: dict[str, Any]
    requirements: dict[str, Any]
    structural_patches: tuple[dict[str, Any], ...]
    targets: tuple[RuntimeManifestTarget, ...]

    @property
    def precision(self) -> str | None:
        """Return the common manifest precision, or ``None`` for mixed."""

        precision = self.requirements.get("precision")
        if precision in SUPPORTED_PRECISIONS:
            return str(precision)
        return None

    @property
    def runtime_precision(self) -> str | None:
        """Return the common native precision, or ``None`` for mixed."""

        precision = self.precision
        if precision is None:
            return None
        return "nvfp4" if precision == "fp4" else precision

    @property
    def rank(self) -> int:
        """Return the common rank declared in requirements."""

        return int(self.requirements.get("rank", 32))


def parse_runtime_manifest(quantization_config: dict[str, Any]) -> RuntimeManifest | None:
    """Parse and validate ``quantization_config.runtime_manifest`` if present."""

    raw = quantization_config.get("runtime_manifest")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("quantization_config.runtime_manifest must be a JSON object.")

    schema = _required(raw, "schema", str)
    if schema != SCHEMA:
        raise ValueError(f"Unsupported runtime_manifest schema {schema!r}; expected {SCHEMA!r}.")

    version = _required(raw, "version", int)
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported runtime_manifest version {version}; expected {SUPPORTED_VERSION}.")

    nunchaku_format_version = _required(raw, "nunchaku_format_version", int)
    if nunchaku_format_version != SUPPORTED_FORMAT_VERSION:
        raise ValueError(
            "Unsupported runtime_manifest nunchaku_format_version "
            f"{nunchaku_format_version}; expected {SUPPORTED_FORMAT_VERSION}."
        )

    component = _required(raw, "component", str)
    producer = _required(raw, "producer", dict)
    requirements = _required(raw, "requirements", dict)
    _validate_producer(producer)
    _validate_requirements(requirements)
    structural_patches = _required(raw, "structural_patches", list)
    targets_raw = _required(raw, "targets", list)
    if not targets_raw:
        raise ValueError("runtime_manifest.targets must contain at least one target.")

    precision = requirements.get("precision")
    if precision not in (*SUPPORTED_PRECISIONS, "mixed"):
        raise ValueError(f"Unsupported runtime_manifest requirements.precision {precision!r}.")

    return RuntimeManifest(
        schema=schema,
        version=version,
        component=component,
        nunchaku_format_version=nunchaku_format_version,
        producer=dict(producer),
        requirements=dict(requirements),
        structural_patches=tuple(_validate_structural_patch(patch) for patch in structural_patches),
        targets=tuple(_parse_target(index, target) for index, target in enumerate(targets_raw)),
    )


def _validate_producer(producer: dict[str, Any]) -> None:
    _required(producer, "name", str)
    _required(producer, "version", str)


def _validate_requirements(requirements: dict[str, Any]) -> None:
    _required(requirements, "method", str)
    _required(requirements, "precision", str)
    _required(requirements, "rank", int)
    _required(requirements, "weight_dtype", str)
    _required(requirements, "activation_dtype", str)
    if "torch_dtype" not in requirements:
        raise ValueError("runtime_manifest requirements missing required field 'torch_dtype'.")


def _parse_target(index: int, raw: Any) -> RuntimeManifestTarget:
    if not isinstance(raw, dict):
        raise ValueError(f"runtime_manifest.targets[{index}] must be a JSON object.")

    checkpoint_prefix = _required(raw, "checkpoint_prefix", str)
    source_modules = _required(raw, "source_modules", list)
    roles = _required(raw, "roles", list)
    kind = _required(raw, "kind", str)
    nunchaku_op = _required(raw, "nunchaku_op", str)
    precision = _required(raw, "precision", str)
    group_size = _required(raw, "group_size", int)
    rank = _required(raw, "rank", int)
    has_bias = _required(raw, "has_bias", bool)
    op_options = _required(raw, "op_options", dict)
    activation = _required(raw, "activation", dict)

    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"Unsupported runtime_manifest target kind {kind!r} at {checkpoint_prefix!r}.")
    if nunchaku_op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported runtime_manifest nunchaku_op {nunchaku_op!r} at {checkpoint_prefix!r}.")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported runtime_manifest target precision {precision!r} at {checkpoint_prefix!r}.")
    if group_size <= 0:
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} must have positive group_size.")
    if rank < 0:
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} must have non-negative rank.")
    if not all(isinstance(item, str) for item in source_modules):
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} source_modules must be strings.")
    if not all(isinstance(item, str) for item in roles):
        raise ValueError(f"runtime_manifest target {checkpoint_prefix!r} roles must be strings.")

    if nunchaku_op == "adanorm_awq_w4a16":
        splits = op_options.get("adanorm_splits")
        if not isinstance(splits, int) or splits <= 0:
            raise ValueError(
                f"runtime_manifest target {checkpoint_prefix!r} requires positive op_options.adanorm_splits."
            )

    return RuntimeManifestTarget(
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        checkpoint_prefix=checkpoint_prefix,
        source_modules=tuple(source_modules),
        roles=tuple(roles),
        kind=kind,
        nunchaku_op=nunchaku_op,
        precision=precision,
        group_size=group_size,
        rank=rank,
        has_bias=has_bias,
        op_options=dict(op_options),
        activation=dict(activation),
    )


def _validate_structural_patch(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("runtime_manifest structural patches must be JSON objects.")
    patch_type = _required(raw, "type", str)
    if patch_type not in {"split_linear_output", "split_linear_input"}:
        raise ValueError(f"Unsupported runtime_manifest structural patch type {patch_type!r}.")
    module = _required(raw, "module", str)
    args = _required(raw, "args", dict)
    splits = args.get("splits")
    if not isinstance(splits, list):
        raise ValueError(f"runtime_manifest structural patch {module!r} requires args.splits list.")
    return {"type": patch_type, "module": module, "args": dict(args)}


def _required(raw: dict[str, Any], key: str, expected_type: type) -> Any:
    if key not in raw:
        raise ValueError(f"runtime_manifest is missing required field {key!r}.")
    value = raw[key]
    if not isinstance(value, expected_type):
        raise ValueError(f"runtime_manifest field {key!r} must be {expected_type.__name__}.")
    return value
