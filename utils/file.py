
import logging
import os
import platform
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def get_file_creation_time(file_path: str) -> str:
    """
    获取数字化时间，格式化为"2026年2月"

    优先级：
      1. 图片所在文件夹的创建时间（birthtime）
      2. 图片文件本身的创建时间（birthtime）
      3. 兜底：当前时间
    """
    image_path = Path(file_path)

    # 优先：文件夹创建时间
    folder_path = image_path.parent
    ts = _get_birthtime(folder_path)
    if ts:
        dt = datetime.fromtimestamp(ts)
        formatted = f"{dt.year}年{dt.month}月"
        logger.info(f"[数字化时间] 文件夹创建时间: {formatted}（{folder_path.name}）")
        return formatted

    # 降级：文件本身创建时间
    ts = _get_birthtime(image_path)
    if ts:
        dt = datetime.fromtimestamp(ts)
        formatted = f"{dt.year}年{dt.month}月"
        logger.info(f"[数字化时间] 文件创建时间: {formatted}（{image_path.name}）")
        return formatted

    # 兜底：当前时间
    logger.warning("[数字化时间] 无法读取创建时间，使用当前时间")
    now = datetime.now()
    return f"{now.year}年{now.month}月"


@lru_cache(maxsize=4096)
def _get_birthtime(path: Path) -> float | None:
    """
    跨平台获取文件或文件夹的创建时间（birthtime）时间戳。

    各平台策略：
      Windows : st_ctime（Windows 上 ctime 即创建时间）
      macOS   : st_birthtime（真正的创建时间）
      Linux   : 先读 os.stat 的 st_birthtime（Python ≥ 3.12 + 新内核 statx）
                取不到再退回 `stat --format=%W` 子进程。
                不回退 mtime —— 修改时间与创建时间语义不同。

    加 lru_cache：批量处理时同一父目录会被多张图重复查询，
    缓存既省 os.stat 也省 Linux 兜底路径的子进程开销。

    返回：float 时间戳；无法获取返回 None（None 也会被缓存）。
    """
    try:
        stat = os.stat(path)
        system = platform.system()

        if system == "Windows":
            return stat.st_ctime

        if system == "Darwin":
            return getattr(stat, "st_birthtime", None)

        # Linux: 先试 statx 暴露的 st_birthtime
        birthtime = getattr(stat, "st_birthtime", None)
        if birthtime is not None:
            return birthtime

        # 兜底：调用 stat 命令读取 %W
        result = subprocess.run(
            ["stat", "--format=%W", str(path)],
            capture_output=True, text=True, timeout=3
        )
        birth_ts = result.stdout.strip()
        if birth_ts and birth_ts not in ("0", "-", ""):
            return float(birth_ts)
        return None

    except Exception as e:
        logger.error(f"[数字化时间] _get_birthtime 失败 ({path}): {e}")
        return None
