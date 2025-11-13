import os
import copy
import datetime
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import yaml


DictLike = Mapping[str, Any]


def _read_yaml_file(path: str) -> Dict[str, Any]:
	"""Load a YAML file, returning an empty dict if the file does not exist."""
	if not path:
		return {}
	if not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError(f"YAML at {path} must define a mapping/object at top level.")
	return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
	"""Recursively merge two dicts: values in override take precedence."""
	result: Dict[str, Any] = copy.deepcopy(base)
	for key, value in (override or {}).items():
		if (
			key in result
			and isinstance(result[key], dict)
			and isinstance(value, dict)
		):
			result[key] = _deep_merge(result[key], value)
		else:
			result[key] = copy.deepcopy(value)
	return result


def _set_deep(config: Dict[str, Any], path: Iterable[str], value: Any) -> None:
	"""Set a deep key in a nested dict given an iterable of keys."""
	curr = config
	parts = list(path)
	for i, key in enumerate(parts):
		is_last = i == len(parts) - 1
		if is_last:
			curr[key] = value
		else:
			if key not in curr or not isinstance(curr[key], dict):
				curr[key] = {}
			curr = curr[key]


def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Apply CLI-style overrides onto the config.
	Supports dot-notation keys, e.g. {'detector.conf_threshold': 0.35}
	"""
	if not overrides:
		return config
	result = copy.deepcopy(config)
	for raw_key, value in overrides.items():
		if raw_key is None:
			continue
		key = str(raw_key)
		path_parts = key.split(".")
		_set_deep(result, path_parts, value)
	return result


def _expand_env_vars(config: Any) -> Any:
	"""Expand environment variables in all string values of the config."""
	if isinstance(config, dict):
		return {k: _expand_env_vars(v) for k, v in config.items()}
	if isinstance(config, list):
		return [_expand_env_vars(v) for v in config]
	if isinstance(config, str):
		return os.path.expandvars(config)
	return config


def _compute_run_dir(config: Dict[str, Any]) -> str:
	"""Compute a run directory from pattern and current timestamp."""
	paths = config.get("paths", {})
	pattern = paths.get("run_dir_pattern", "runs/track_{timestamp}")
	now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = pattern.format(timestamp=now)
	# Prepend outputs_dir if pattern is not absolute and outputs_dir exists
	outputs_dir = paths.get("outputs_dir")
	if outputs_dir and not os.path.isabs(run_dir):
		run_dir = os.path.join(outputs_dir, run_dir)
	return run_dir


def load_config(
	default_yaml_path: str = "configs/default.yaml",
	detector_yaml_path: Optional[str] = "configs/detector.yaml",
	tracker_yaml_path: Optional[str] = "configs/tracker.yaml",
	runtime_yaml_path: Optional[str] = "configs/runtime.yaml",
	cli_overrides: Optional[Union[Dict[str, Any], "argparse.Namespace"]] = None,
) -> Dict[str, Any]:

	# Base: default.yaml
	default_cfg = _read_yaml_file(default_yaml_path)

	# Discover include paths (unless explicitly provided)
	includes = (default_cfg or {}).get("includes", {}) if isinstance(default_cfg, dict) else {}
	detector_yaml_path = detector_yaml_path or includes.get("detector")
	tracker_yaml_path = tracker_yaml_path or includes.get("tracker")
	runtime_yaml_path = runtime_yaml_path or includes.get("runtime")

	# Merge in order
	merged = default_cfg
	merged = _deep_merge(merged, _read_yaml_file(detector_yaml_path) if detector_yaml_path else {})
	merged = _deep_merge(merged, _read_yaml_file(tracker_yaml_path) if tracker_yaml_path else {})
	merged = _deep_merge(merged, _read_yaml_file(runtime_yaml_path) if runtime_yaml_path else {})

	# Normalize CLI overrides input
	if cli_overrides is not None and not isinstance(cli_overrides, dict):
		try:
			# e.g., argparse.Namespace
			cli_overrides = {k: getattr(cli_overrides, k) for k in vars(cli_overrides)}
		except Exception:
			raise TypeError("cli_overrides must be a dict or argparse.Namespace-like object.")

	# Drop None overrides to avoid erasing config unintentionally
	non_null_overrides: Dict[str, Any] = {}
	for k, v in (cli_overrides or {}).items():
		if v is not None:
			non_null_overrides[k] = v

	# Apply CLI overrides
	merged = _apply_overrides(merged, non_null_overrides)

	# Expand env vars
	merged = _expand_env_vars(merged)

	# Ensure runtime.resolved.run_dir is computed if not provided
	runtime_cfg = merged.setdefault("resolved", {})  # backward-compatible location
	# If project keeps resolved under runtime.resolved, mirror it
	if "runtime" in merged and isinstance(merged["runtime"], dict):
		merged["runtime"].setdefault("resolved", {})
		runtime_cfg = merged["runtime"]["resolved"]

	if runtime_cfg.get("run_dir") in (None, "", "null"):
		runtime_cfg["run_dir"] = _compute_run_dir(merged)

	return merged


__all__ = [
	"load_config",
]


