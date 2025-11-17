import os
import torch
from typing import Any, Dict, Optional, Tuple



def _preferred_device_from_config(config: Dict[str, Any]) -> str:
    pref = (config or {}).get("device", "auto")
    return str(pref).lower()


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:

    preferred = _preferred_device_from_config(config or {})
    if preferred in ("cuda", "gpu"):
        # Check both CUDA availability AND that devices are actually accessible
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.device("cuda")
        # fallback
        return torch.device("cpu")
    if preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preferred == "cpu":
        return torch.device("cpu")

    # auto
    # Check both CUDA availability AND that devices are actually accessible
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_environment(config: Dict[str, Any], device: Optional[torch.device] = None) -> torch.device:
    """
    Apply basic torch runtime settings (threads, cudnn flags) and return the resolved device.
    """
    device = device or get_device(config)

    # Threads
    num_threads = (config or {}).get("torch_num_threads")
    if isinstance(num_threads, int) and num_threads > 0:
        try:
            torch.set_num_threads(num_threads)
        except Exception:
            pass

    # Determinism vs benchmark
    # Accept both 'deterministic' and the misspelled 'determenistic' from configs.
    deterministic = (config or {}).get("deterministic")
    if deterministic is None:
        deterministic = (config or {}).get("determenistic")
    deterministic = bool(deterministic) if deterministic is not None else False

    cudnn_benchmark = bool((config or {}).get("cudnn_benchmark", not deterministic))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = cudnn_benchmark
        # torch.use_deterministic_algorithms is more strict; only enable if requested
        try:
            torch.use_deterministic_algorithms(deterministic)
        except Exception:
            pass

    return device


def should_use_amp(module_cfg: Optional[Dict[str, Any]], global_cfg: Optional[Dict[str, Any]], device: torch.device) -> bool:
    """
    Decide whether to enable autocast (AMP) for the current module.
    Priority:
      1) module_cfg['amp'] if set (bool)
      2) global_cfg['precision'] == 'amp'
    Only enabled on non-CPU devices that support autocast.
    """
    # Module override
    if isinstance(module_cfg, dict) and "amp" in module_cfg:
        requested = bool(module_cfg.get("amp"))
    else:
        requested = str((global_cfg or {}).get("precision", "fp32")).lower() == "amp"

    if not requested:
        return False

    if device.type == "cuda":
        return True
    # MPS autocast is available in recent torch; keep conservative enablement
    if device.type == "mps":
        # Allow AMP on MPS if explicitly requested at module level
        if isinstance(module_cfg, dict) and "amp" in module_cfg and bool(module_cfg.get("amp")):
            return True
        return False
    return False


def should_use_half(module_cfg: Optional[Dict[str, Any]], global_cfg: Optional[Dict[str, Any]], device: torch.device) -> bool:
    """
    Decide whether to use FP16 weights/inputs ("half" mode).
    Half typically applies only on CUDA; not recommended on CPU/MPS.
    Priority:
      1) module_cfg['half'] if set
      2) global_cfg['half']
    """
    if isinstance(module_cfg, dict) and "half" in module_cfg:
        requested = bool(module_cfg.get("half"))
    else:
        requested = bool((global_cfg or {}).get("half", False))
    if not requested:
        return False
    return device.type == "cuda"


def amp_autocast_kwargs(device: torch.device) -> Dict[str, Any]:
    """
    Provide dtype/device_type for torch.autocast context manager based on device.
    Usage:
        with torch.autocast(**amp_autocast_kwargs(device)):
            ...
    """
    if device.type == "cuda":
        return {"device_type": "cuda", "dtype": torch.float16}
    if device.type == "mps":
        # autocast on MPS currently uses float16
        return {"device_type": "mps", "dtype": torch.float16}
    # CPU: no-op (caller should check should_use_amp first)
    return {"device_type": "cpu"}

