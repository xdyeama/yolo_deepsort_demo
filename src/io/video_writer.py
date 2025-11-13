from typing import Optional, Tuple
import os

import cv2
import numpy as np


class VideoWriter:
	"""
	OpenCV-based video writer that:
	- Initializes once with frame size and FPS (or lazily from first frame)
	- Writes frames only if enabled
	- Resizes frames to target size if they don't match
	"""

	def __init__(
		self,
		output_path: str,
		fps: Optional[float],
		frame_size: Optional[Tuple[int, int]],
		fourcc: str = "mp4v",
		enabled: bool = True,
		is_color: bool = True,
	):
		self.output_path = output_path
		self.requested_fps = fps
		self.requested_size = frame_size  # (width, height)
		self.fourcc_str = fourcc
		self.enabled = bool(enabled)
		self.is_color = is_color

		self._writer: Optional[cv2.VideoWriter] = None
		self._fps: Optional[float] = None
		self._size: Optional[Tuple[int, int]] = None

	def _ensure_dir(self) -> None:
		if not self.output_path:
			return
		parent = os.path.dirname(self.output_path)
		if parent and not os.path.exists(parent):
			os.makedirs(parent, exist_ok=True)

	def _open(self, size_from_frame: Optional[Tuple[int, int]] = None, fps_from_frame: Optional[float] = None) -> None:
		if not self.enabled:
			return
		if self._writer is not None:
			return

		size = self.requested_size or size_from_frame
		if not size or size[0] <= 0 or size[1] <= 0:
			raise ValueError("VideoWriter needs a valid frame_size (width, height).")

		fps = self.requested_fps or fps_from_frame or 30.0
		if fps <= 0:
			fps = 30.0

		self._ensure_dir()
		fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
		writer = cv2.VideoWriter(self.output_path, fourcc, float(fps), (int(size[0]), int(size[1])), self.is_color)
		if not writer or not writer.isOpened():
			raise RuntimeError(f"Failed to open VideoWriter for path: {self.output_path}")

		self._writer = writer
		self._fps = float(fps)
		self._size = (int(size[0]), int(size[1]))

	@property
	def is_open(self) -> bool:
		return self._writer is not None

	@property
	def fps(self) -> Optional[float]:
		return self._fps

	@property
	def frame_size(self) -> Optional[Tuple[int, int]]:
		return self._size

	def write(self, frame_bgr: np.ndarray) -> None:
		"""
		Write a frame to the video file. Lazily initializes writer if needed.
		"""
		if not self.enabled:
			return

		if frame_bgr is None or frame_bgr.size == 0:
			return

		h, w = frame_bgr.shape[:2]
		if self._writer is None:
			self._open(size_from_frame=(w, h))

		assert self._writer is not None
		target_w, target_h = self._size  # type: ignore[misc]
		if (w, h) != (target_w, target_h):
			frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
		self._writer.write(frame_bgr)

	def release(self) -> None:
		if self._writer is not None:
			try:
				self._writer.release()
			finally:
				self._writer = None

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, tb):
		self.release()
		return False

import cv2