from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    video_path: Path
    median_ksize: int = 7
    mask_threshold: int = 127
    signal_threshold: int = 15
    white_clip: int = 200
    min_contour_area: float = 20.0
    mog2_history: int = 300
    mog2_var_threshold: float = 24.0
    mog2_shadows: bool = False
    display_size: tuple[int, int] = (640, 480)
    headless: bool = False
    output_path: Path = Path("out1.mp4")
