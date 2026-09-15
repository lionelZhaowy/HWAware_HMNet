"""Build a local QONNX wheel using the existing GPU ONNX Runtime distribution.

Only the version and runtime requirement change; all qonnx code is preserved.
The input is the official PyPI qonnx 1.0.0 wheel, verified by SHA256.
Requires the already installed `wheel` package. Does not install anything.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


UPSTREAM_SHA256 = '6454a935d5e269ccd5697078bf1bec27e3bcb5adeba42d4dad5921ea83832ce4'
LOCAL_VERSION = '1.0.0+ortgpu'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream_wheel', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    wheel = args.upstream_wheel.resolve()
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('Input does not match the official qonnx 1.0.0 wheel')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qonnx-ortgpu-') as temporary:
        subprocess.run([sys.executable, '-m', 'wheel', 'unpack', str(wheel),
                        '--dest', temporary], check=True)
        unpacked = Path(temporary) / 'qonnx-1.0.0'
        info = unpacked / 'qonnx-1.0.0.dist-info'
        metadata = info / 'METADATA'
        text = metadata.read_text()
        for before, after in [
            ('Version: 1.0.0\n', f'Version: {LOCAL_VERSION}\n'),
            ('Requires-Dist: onnxruntime>=1.16.1\n',
             'Requires-Dist: onnxruntime-gpu>=1.16.1\n'),
        ]:
            if text.count(before) != 1:
                raise ValueError(f'Unexpected upstream metadata: {before!r}')
            text = text.replace(before, after)
        metadata.write_text(text)
        info.rename(unpacked / f'qonnx-{LOCAL_VERSION}.dist-info')
        # wheel pack regenerates RECORD, including hashes and sizes.
        subprocess.run([sys.executable, '-m', 'wheel', 'pack', str(unpacked),
                        '--dest-dir', str(output)], check=True)
    result = output / f'qonnx-{LOCAL_VERSION}-py2.py3-none-any.whl'
    with zipfile.ZipFile(wheel) as original, zipfile.ZipFile(result) as patched:
        code_files = {name for name in original.namelist() if name.startswith('qonnx/')}
        if code_files != {name for name in patched.namelist() if name.startswith('qonnx/')}:
            raise RuntimeError('QONNX payload file list changed')
        for name in code_files:
            if original.read(name) != patched.read(name):
                raise RuntimeError(f'QONNX payload changed: {name}')
    print(f'Verified {len(code_files)} unchanged QONNX payload files: {result}')
    print('SHA256:', hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
