"""cl2cstring (src/cl2cstring.c) embeds each kernel source in the build as a C string: every character of the
.cl file must reach the compiled kernel unchanged.

It used to copy only the characters 0-125, so '~' (126) was silently dropped: `i & ~1` compiled as `i & 1`.

Usage: SEISCL_BIN=<build dir holding cl2cstring> python SeisCL/tests/test_cl2cstring.py
"""
import os
import subprocess
import sys
import tempfile


def unescape_c_string(body):
    """The characters of a C string literal body written by cl2cstring."""
    esc = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v",
           "\\": "\\", "\"": "\"", "'": "'", "?": "?", "0": "\0"}
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c == "\\":
            out.append(esc[body[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def test_printable_ascii_survives(bindir):
    src = "".join(chr(c) for c in range(32, 127)) + "\n\tx = i & ~1;\n"
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "probe.cl"), "w") as f:
            f.write(src)
        subprocess.run([os.path.join(bindir, "cl2cstring"), "probe.cl", "probe.hcl"], cwd=d, check=True)
        text = open(os.path.join(d, "probe.hcl")).read()
    start = text.index("= \"") + 3
    end = text.rindex("\";")
    got = unescape_c_string(text[start:end])
    missing = sorted({c for c in src if c not in got})
    ok = got == src
    print(f"{'PASS' if ok else 'FAIL'}  cl2cstring keeps every printable ASCII character "
          f"(missing: {[repr(c) for c in missing]}, threshold: none)")
    return ok


if __name__ == "__main__":
    bindir = os.environ.get("SEISCL_BIN")
    if not bindir:
        sys.exit("set SEISCL_BIN to the build directory that holds cl2cstring")
    sys.exit(0 if test_printable_ascii_survives(bindir) else 1)
