
import logging
import os
import platform
import subprocess
from datetime import datetime
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


def _get_birthtime(path: Path) -> float | None:
    """
    跨平台获取文件或文件夹的创建时间（birthtime）时间戳。

    各平台策略：
      Windows : st_ctime（Windows上ctime就是创建时间）
      macOS   : st_birthtime（真正的创建时间）
      Linux   : stat --format=%W（部分文件系统支持）→ 不支持则返回None
                注意：Linux不回退mtime，因为mtime语义是修改时间而非创建时间，
                      回退mtime会导致数字化时间语义错误。

    返回：float时间戳，无法获取返回None
    """
    try:
        stat = os.stat(path)
        system = platform.system()

        if system == "Windows":
            return stat.st_ctime

        elif system == "Darwin":
            if hasattr(stat, 'st_birthtime'):
                return stat.st_birthtime
            return None

        else:
            # Linux: 尝试 stat --format=%W（Birth时间）
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
