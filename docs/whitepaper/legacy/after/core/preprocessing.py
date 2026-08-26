import cv2
import numpy as np

from core.config import Config


def preprocess(frame: np.ndarray, ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, ksize)


def static_mask(reference_frame: np.ndarray, threshold: int) -> np.ndarray:
    """Binary mask marking the (dark) region of interest in a reference frame."""
    _, mask = cv2.threshold(reference_frame, threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def segment(frame: np.ndarray, mask: np.ndarray, cfg: Config) -> np.ndarray:
    masked = np.where(mask == 255, frame, 0).astype(np.uint8)
    _, signal_mask = cv2.threshold(masked, cfg.signal_threshold, 255, cv2.THRESH_BINARY)
    masked = np.where(signal_mask == 255, masked, 0)
    return np.where(masked > cfg.white_clip, 0, masked).astype(np.uint8)


def find_contours(frame: np.ndarray, min_area: float, output: np.ndarray) -> np.ndarray:
    """Draw the frame's contours (above min_area) in color, returning a BGR image."""
    _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(output, contours, -1, (161, 177, 247), 2)
