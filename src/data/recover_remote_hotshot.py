"""HTTP Range定向恢复Hotshot成员；不下载7.9GB训练包的无关视频。"""

import concurrent.futures
import hashlib
import io
import json
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path
import requests
from artifacts import paper_json


class RemoteArchive(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        r = requests.get(url, headers={"Range": "bytes=-131072"}, timeout=(15, 60))
        r.raise_for_status()
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
        if r.status_code != 206 or not match:
            raise ValueError("服务端不支持精确Range")
        self.size = int(match[3])
        self.url = r.url

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        )
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = self.fetch(self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data

    def fetch(self, start, end):
        end = min(end, self.size - 1)
        for attempt in range(4):
            try:
                r = requests.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 90)
                )
                r.raise_for_status()
                if (
                    r.status_code != 206
                    or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                ):
                    raise ValueError("Range回复不对应请求，拒绝使用错误缓存")
                if len(r.content) != end - start + 1:
                    raise ValueError("Range长度不符")
                return r.content
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)


def main():
    root = Path.cwd()
    base = root / "datasets/recovery"
    metadata = json.loads((base / "metadata/videofeedback_train.json").read_text())
    wanted = {
        str(r["id"])
        for r in metadata
        if "vidprom_hotshot/" in r["video link"] and r["dynamic degree"] >= 3
    }
    remote = RemoteArchive(
        "https://hf-mirror.com/datasets/TIGER-Lab/VideoFeedback/resolve/main/train/videos_annotated.zip"
    )
    with zipfile.ZipFile(remote) as archive:
        members = [
            m
            for m in archive.infolist()
            if Path(m.filename).suffix == ".mp4" and Path(m.filename).stem in wanted
        ]
    if len(members) != len(wanted):
        raise ValueError("训练归档与元数据目标身份不全")
    print(f"HTTP Range: {len(members)} target members / archive {remote.size} bytes", flush=True)

    def restore(m):
        # ZIP本地头中的文件名和extra各最多65535字节；一次Range覆盖两者及压缩数据。
        blob = remote.fetch(m.header_offset, m.header_offset + 30 + 131070 + m.compress_size - 1)
        header = struct.unpack("<4s5H3I2H", blob[:30])
        if header[0] != b"PK\x03\x04" or header[3] != m.compress_type:
            raise ValueError("ZIP成员头不符")
        offset = 30 + header[-2] + header[-1]
        compressed = blob[offset : offset + m.compress_size]
        if m.compress_type == zipfile.ZIP_STORED:
            data = compressed
        elif m.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(compressed, -15)
        else:
            raise ValueError("不支持的成员压缩方式")
        if len(data) != m.file_size or zlib.crc32(data) & 0xFFFFFFFF != m.CRC:
            raise ValueError("成员CRC/尺寸不符")
        digest = hashlib.sha256(data).hexdigest()
        target = root / "datasets/videofeedback/fake/Hotshot-XL" / Path(m.filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("已有视频不同")
        else:
            temp = target.with_suffix(".recovering")
            temp.write_bytes(data)
            temp.replace(target)
        return dict(
            path=str(target.relative_to(root)),
            member=m.filename,
            bytes=len(data),
            sha256=digest,
            crc=m.CRC,
        )

    records = []
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for record in pool.map(restore, members):
            records.append(record)
            if len(records) % 32 == 0:
                paper_json(
                    base / "remote_train_progress.json",
                    dict(
                        completed=len(records), total=len(members), elapsed=time.monotonic() - start
                    ),
                )
                print(f"[range restore] {len(records)}/{len(members)}", flush=True)
    paper_json(
        base / "receipts/train.json",
        dict(
            status="completed",
            archive_bytes=remote.size,
            mode="HTTP Range / member CRC verified",
            videos=records,
        ),
    )
    print(f"completed {len(records)}", flush=True)


if __name__ == "__main__":
    main()
