"""
Точка входа для DDP (torchrun) из пайплайна.

На Windows-сборках PyTorch 2.4+ TCPStore по умолчанию use_libuv=True, но libuv в wheel
не собран (pytorch/pytorch#139990). Официальный обход: USE_LIBUV=0 до import torch
(см. docs.pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html).

Если env недостаточно (elastic rendezvous), тот же приём, что в DeepSpeed #7064 и
pytorch#148266: TCPStore(..., use_libuv=False) на Windows.
"""
from __future__ import annotations

import os
import sys

os.environ["USE_LIBUV"] = "0"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")


def _apply_windows_tcpstore_compat() -> None:
    if sys.platform != "win32":
        return
    import torch.distributed as dist

    if getattr(dist.TCPStore, "_diploma_tcpstore_compat", False):
        return

    _orig = dist.TCPStore

    class _WindowsTCPStore(_orig):
        def __init__(self, *args, **kwargs):
            kwargs["use_libuv"] = False
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                kwargs.pop("use_libuv", None)
                super().__init__(*args, **kwargs)

    _WindowsTCPStore._diploma_tcpstore_compat = True  # type: ignore[attr-defined]
    dist.TCPStore = _WindowsTCPStore


def main() -> None:
    _apply_windows_tcpstore_compat()
    from torch.distributed.run import main as torchrun_main

    sys.argv[0] = "torchrun"
    torchrun_main()


if __name__ == "__main__":
    main()
