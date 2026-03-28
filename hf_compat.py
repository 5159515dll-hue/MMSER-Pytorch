"""补齐旧版 torch 与新版 transformers 之间的兼容层。

部分服务器上的全局 torch 只暴露 `_register_pytree_node`，而新版 transformers
期望存在 `register_pytree_node`。这里负责在导入 transformers 之前补齐接口。
"""

from __future__ import annotations

from typing import Any, Callable


def ensure_transformers_torch_compat() -> None:
    """Backfill the pytree API expected by newer transformers releases."""

    try:
        from torch.utils import _pytree
    except Exception:
        return

    if hasattr(_pytree, "register_pytree_node"):
        return

    legacy_register = getattr(_pytree, "_register_pytree_node", None)
    if legacy_register is None:
        return

    def _register_pytree_node(
        cls: type[Any],
        flatten_fn: Callable[..., Any],
        unflatten_fn: Callable[..., Any],
        *,
        serialized_type_name: str | None = None,
        to_dumpable_context: Callable[..., Any] | None = None,
        from_dumpable_context: Callable[..., Any] | None = None,
        flatten_with_keys_fn: Callable[..., Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if serialized_type_name is not None:
            kwargs["serialized_type_name"] = serialized_type_name
        if to_dumpable_context is not None:
            kwargs["to_dumpable_context"] = to_dumpable_context
        if from_dumpable_context is not None:
            kwargs["from_dumpable_context"] = from_dumpable_context
        if flatten_with_keys_fn is not None:
            kwargs["flatten_with_keys_fn"] = flatten_with_keys_fn
        try:
            legacy_register(cls, flatten_fn, unflatten_fn, **kwargs)
        except TypeError:
            legacy_register(cls, flatten_fn, unflatten_fn)

    _pytree.register_pytree_node = _register_pytree_node
