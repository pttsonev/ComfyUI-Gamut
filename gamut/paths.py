"""Pure EXR filename formatting and explicit sequence expansion, without ComfyUI."""

import os
from pathlib import Path
import re


PATH_MODES = ("versioned", "hq", "filename")
VERSION_LAYOUTS = ("suffix", "directory", "none")
_READ_TOKEN = re.compile(r"\{frame\}|#+|%0([1-9][0-9]*)d")


def integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def frame_path(folder, name, extension, frame, frame_pad, version, version_layout="suffix"):
    suffix = f"_v{version:03d}" if version_layout == "suffix" else ""
    return Path(folder) / f"{name}{suffix}.{frame:0{frame_pad}d}.{extension}"


def prepare_output(prefix, extension, count=1, start_frame=1001, frame_pad=4, version=1,
                   path_mode="versioned", version_layout="suffix"):
    """Return a prefix with its version directory/tokens expanded, keeping {frame}."""
    prefix = os.fspath(prefix)
    if not prefix or prefix.endswith(("/", "\\")) or "\x00" in prefix:
        raise ValueError("Supply a filename prefix, not an empty name or directory.")
    if os.path.basename(prefix) in ("", ".", ".."):
        raise ValueError("Supply a filename prefix, not a directory.")
    if os.path.splitdrive(prefix)[0] and not os.path.isabs(prefix):
        raise ValueError("A drive-relative path is ambiguous; supply an absolute path.")
    if path_mode not in PATH_MODES or version_layout not in VERSION_LAYOUTS:
        raise ValueError("Unknown path_mode or version_layout.")
    integer(count, "frame count", 1)
    integer(start_frame, "start_frame", 0)
    integer(frame_pad, "frame_pad", 1)
    integer(version, "version", 1)
    if path_mode == "filename":
        rest = prefix.replace("{frame}", "").replace("{version}", "")
        if "{" in rest or "}" in rest or prefix.count("{frame}") > 1 or prefix.count("{version}") > 1:
            raise ValueError("Filename accepts at most one {frame} and one {version}; unknown tokens are refused.")
        if "{frame}" in os.path.dirname(prefix):
            raise ValueError("The {frame} token must be in the filename, not a directory.")
        if count > 1 and "{frame}" not in prefix:
            raise ValueError("An exact filename requires one frame; use {frame} for a batch.")
        if Path(prefix).suffix.lower() != f".{extension}":
            raise ValueError(f"Filename mode requires a .{extension} extension.")
        return prefix.replace("{version}", f"v{version:03d}")
    # HQ relative uses only ComfyUI's counter, ignoring version/frame controls.
    if version_layout == "directory" and not (path_mode == "hq" and not os.path.isabs(prefix)):
        path = Path(prefix)
        return str(path.parent / f"v{version:03d}" / path.name)
    return prefix


def output_paths(folder, name, extension, count=1, start_frame=1001, frame_pad=4, version=1,
                 path_mode="versioned", version_layout="suffix", *, absolute=True, counter=1):
    """Format already-resolved targets. No files or counters are reserved here."""
    if path_mode == "filename":
        result = [Path(folder) / name.replace("{frame}", f"{start_frame + i:0{frame_pad}d}")
                  for i in range(count)]
    elif path_mode == "hq" and not absolute:
        result = [Path(folder) / f"{name}_{counter + i:05d}_.{extension}" for i in range(count)]
    else:
        result = [frame_path(folder, name, extension, start_frame + i, frame_pad, version,
                             version_layout) for i in range(count)]
    if len({os.path.normcase(os.path.abspath(path)) for path in result}) != len(result):
        raise ValueError("Output template produces duplicate destinations.")
    return result


def sequence_paths(path, start_frame=1001, frame_count=1, frame_step=1, frame_pad=4):
    """Expand one explicit frame pattern; never glob or silently skip missing frames."""
    integer(start_frame, "start_frame", 0)
    integer(frame_count, "frame_count", 1)
    integer(frame_step, "frame_step", 1)
    integer(frame_pad, "frame_pad", 1)
    path = os.fspath(path)
    if not path or "\x00" in path or path.endswith(("/", "\\")):
        raise ValueError("Supply an EXR file or frame pattern.")
    tokens = list(_READ_TOKEN.finditer(path))
    rest = _READ_TOKEN.sub("", path)
    if len(tokens) > 1 or any(char in rest for char in "{}#%"):
        raise ValueError("Use one {frame}, hash run or %0Nd sequence token, never mixed patterns.")
    if not tokens:
        if frame_count != 1:
            raise ValueError("An exact file requires frame_count=1; supply a sequence pattern.")
        return [path]
    token = tokens[0]
    if token.start() < max(path.rfind("/"), path.rfind("\\")) + 1:
        raise ValueError("The sequence token must be in the filename, not a directory.")
    text = token.group()
    pad = len(text) if text.startswith("#") else int(token.group(1)) if text.startswith("%") else frame_pad
    if pad > 12:
        raise ValueError("Sequence pattern padding must be <= 12.")
    return [path[:token.start()] + f"{start_frame + i * frame_step:0{pad}d}" + path[token.end():]
            for i in range(frame_count)]


def fingerprint(files):
    result = []
    for filename in files:
        path = Path(filename).absolute()
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except FileNotFoundError:
            result.append((str(path), "missing"))
    return tuple(result)
