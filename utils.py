import os
import shutil
import urllib.request
import tarfile
from typing import Any, Callable, Optional, Union, Dict
import time
import io
from webdataset.writer import make_encoder, gopen

def download(url, out_file=None):
    """Download a file from url and save it in the current directory."""
    # Get the file name from the url.
    if out_file is None:
        file_name = url.split("/")[-1]
    else:
        file_name = out_file
    if os.path.exists(file_name):
        return file_name
    # Open the url.
    with urllib.request.urlopen(url) as response, open(file_name, "wb") as out_file:
        # Read the file in chunks.
        shutil.copyfileobj(response, out_file)
    print("Downloaded file: {}".format(file_name))
    return file_name


class TarWriter:
    """A class for writing dictionaries to tar files.

    :param fileobj: fileobj: file name for tar file (.tgz/.tar) or open file descriptor
    :param encoder: sample encoding (Default value = True)
    :param compress:  (Default value = None)

    `True` will use an encoder that behaves similar to the automatic
    decoder for `Dataset`. `False` disables encoding and expects byte strings
    (except for metadata, which must be strings). The `encoder` argument can
    also be a `callable`, or a dictionary mapping extensions to encoders.

    The following code will add two file to the tar archive: `a/b.png` and
    `a/b.output.png`.

    ```Python
        tarwriter = TarWriter(stream)
        image = imread("b.jpg")
        image2 = imread("b.out.jpg")
        sample = {"__key__": "a/b", "png": image, "output.png": image2}
        tarwriter.write(sample)
    ```
    """

    def __init__(
        self,
        fileobj,
        user: str = "bigdata",
        group: str = "bigdata",
        mode: int = 0o0444,
        compress: Optional[bool] = None,
        encoder: Union[None, bool, Callable] = True,
        keep_meta: bool = False,
        mtime: Optional[float] = None,
        format: Any = None,
        append=False,
    ):  # sourcery skip: avoid-builtin-shadow
        """Create a tar writer.

        :param fileobj: stream to write data to
        :param user: user for tar files
        :param group: group for tar files
        :param mode: mode for tar files
        :param compress: desired compression
        :param encoder: encoder function
        :param keep_meta: keep metadata (entries starting with "_")
        :param mtime: modification time (set this to some fixed value to get reproducible tar files)
        """
        format = getattr(tarfile, format, format) if format else tarfile.USTAR_FORMAT
        self.mtime = mtime
        m = "w" if append is False else "a"
        if isinstance(fileobj, str):
            if compress is False:
                tarmode = f"{m}|"
            elif compress is True:
                tarmode = f"{m}|gz"
            else:
                tarmode = f"{m}|gz" if fileobj.endswith("gz") else f"{m}|"
            fileobj = gopen(fileobj, "wb")
            self.own_fileobj = fileobj
        else:
            tarmode = f"{m}|gz" if compress is True else f"{m}|"
            self.own_fileobj = None
        self.encoder = make_encoder(encoder)
        self.keep_meta = keep_meta
        self.stream = fileobj
        self.tarstream = tarfile.open(fileobj=fileobj, mode=tarmode)

        self.user = user
        self.group = group
        self.mode = mode
        self.compress = compress

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.close()

    def close(self):
        """Close the tar file."""
        self.tarstream.close()
        if self.own_fileobj is not None:
            self.own_fileobj.close()
            self.own_fileobj = None

    def write(self, obj):
        """Write a dictionary to the tar file.

        :param obj: dictionary of objects to be stored
        :returns: size of the entry

        """
        total = 0
        obj = self.encoder(obj)
        if "__key__" not in obj:
            raise ValueError("object must contain a __key__")
        for k, v in list(obj.items()):
            if k[0] == "_":
                continue
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"{k} doesn't map to a bytes after encoding ({type(v)})"
                )
        key = obj["__key__"]
        for k in sorted(obj.keys()):
            if k == "__key__":
                continue
            if not self.keep_meta and k[0] == "_":
                continue
            v = obj[k]
            if isinstance(v, str):
                v = v.encode("utf-8")
            now = time.time()
            ti = tarfile.TarInfo(key + "." + k)
            ti.size = len(v)
            ti.mtime = self.mtime if self.mtime is not None else now
            ti.mode = self.mode
            ti.uname = self.user
            ti.gname = self.group
            if not isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(f"converter didn't yield bytes: {k}, {type(v)}")
            stream = io.BytesIO(v)
            self.tarstream.addfile(ti, stream)
            total += ti.size
        return total
