from typing import Any, Dict, Generator, Optional, Tuple, Union
import os
import time

import cv2
import numpy as np


def _is_rtsp_url(source: str) -> bool:
	low = source.lower()
	return low.startswith("rtsp://") or low.startswith("rtsps://")


def _infer_input_type(source: Union[int, str]) -> str:
	if isinstance(source, int):
		return "webcam"
	if isinstance(source, str):
		if _is_rtsp_url(source):
			return "rtsp"
		if os.path.exists(source):
			return "file"
		# numeric string means webcam index
		if source.isdigit():
			return "webcam"
	return "file"


class VideoReader:
	"""
	Thin wrapper around OpenCV VideoCapture supporting:
	- file path
	- webcam index
	- RTSP URL

	Yields frames as dicts:
	  { "index": int, "timestamp": float (seconds), "frame": np.ndarray }

	Properties:
	- fps (float or None if unknown)
	- frame_size (Tuple[int, int] -> (width, height))
	"""

	def __init__(self, source: Union[int, str], input_type: str = "auto"):
		self.source = source
		self.input_type = _infer_input_type(source) if input_type == "auto" else input_type
		self._cap: Optional[cv2.VideoCapture] = None
		self._fps: Optional[float] = None
		self._size: Tuple[int, int] = (0, 0)
		self._open()

	def _open(self) -> None:
		if self.input_type == "webcam":
			index = int(self.source) if isinstance(self.source, str) else int(self.source)
			self._cap = cv2.VideoCapture(index)
		else:
			self._cap = cv2.VideoCapture(self.source)

		if not self._cap or not self._cap.isOpened():
			raise RuntimeError(f"Could not open video source: {self.source}")

		# FPS might be 0 or NaN for some sources (e.g., webcams/rtsp)
		fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
		self._fps = fps if fps > 1e-6 else None

		w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		self._size = (w, h)

	def release(self) -> None:
		if self._cap is not None:
			try:
				self._cap.release()
			finally:
				self._cap = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.release()
		# Do not suppress exceptions
		return False

	@property
	def fps(self) -> Optional[float]:
		return self._fps

	@property
	def frame_size(self) -> Tuple[int, int]:
		return self._size

	def frames(self) -> Generator[Dict[str, Any], None, None]:
		"""
		Iterate over frames. Each yield is:
		  { "index": i, "timestamp": ts_seconds, "frame": BGR ndarray }
		Timestamps strategy:
		  - Prefer CAP_PROP_POS_MSEC if available (> 0)
		  - Else if FPS known: index / fps
		  - Else: monotonic time offset from first frame
		"""
		if self._cap is None:
			self._open()

		assert self._cap is not None

		index = 0
		start_monotonic = time.monotonic()
		first_ts: Optional[float] = None

		while True:
			ret, frame = self._cap.read()
			if not ret or frame is None:
				break

			# Try to get timestamp from the capture (milliseconds)
			pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
			if pos_msec > 0.0:
				ts = pos_msec / 1000.0
			elif self._fps:
				ts = index / float(self._fps)
			else:
				now = time.monotonic()
				ts = now - start_monotonic
				if first_ts is None:
					first_ts = ts
				ts = ts - first_ts

			yield {"index": index, "timestamp": ts, "frame": frame}
			index += 1
