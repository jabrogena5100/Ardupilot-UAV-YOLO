import io, urllib.request, collections
class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a small block cache. Counts bytes transferred."""
    def __init__(self, url, block=4 << 20):
        self.url, self.block, self.pos, self.cache, self.transferred, self.requests = url, block, 0, collections.OrderedDict(), 0, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "range-inspect/1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", ""); self.size = int(cr.split("/")[-1]); self.final = r.geturl(); r.read()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def _blk(self, i):
        if i in self.cache:
            self.cache.move_to_end(i); return self.cache[i]
        a, b = i * self.block, min((i + 1) * self.block, self.size) - 1
        req = urllib.request.Request(self.final, headers={"Range": f"bytes={a}-{b}", "User-Agent": "range-inspect/1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = r.read()
        self.requests += 1; self.transferred += len(d); self.cache[i] = d
        if len(self.cache) > 16: self.cache.popitem(last=False)
        return d
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        out = []
        while n > 0 and self.pos < self.size:
            i, o = divmod(self.pos, self.block); d = self._blk(i)[o:o + n]
            if not d: break
            out.append(d); self.pos += len(d); n -= len(d)
        return b"".join(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
