"""Set up and reproduce the frozen compressed Shared Adam study.

Standard-library-only bootstrap: no torch import until the checked environment
starts. Data are separate from code. Never execute anything from an archive or
overwrite a different existing input. No cloud instance is started here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener
import venv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports/federated_shared_optimizer/reproduction-inputs-v1.json"
ENV = ROOT / ".venv-shared-adam"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_path(root, name):
    """Archives may populate only normal relative paths under outputs/."""
    p = PurePosixPath(name)
    require(not p.is_absolute() and len(p.parts) > 1 and p.parts[0] == "outputs"
            and all(x not in (".", "..") for x in p.parts)
            and "\\" not in name and ":" not in name and str(p) == name,
            f"Unsafe input path: {name}")
    root = Path(root).resolve()
    destination = root.joinpath(*p.parts)
    require(not any(q.is_symlink() for q in (destination, *destination.parents) if q != root),
            f"Symlink in input destination: {name}")
    require(destination.resolve().is_relative_to(root), "Input escapes project")
    require(all(not q.exists() or q.is_dir() for q in destination.parents),
            f"Non-directory parent in input destination: {name}")
    return destination


def verify_file(path, spec):
    require(Path(path).is_file() and Path(path).stat().st_size == spec["bytes"]
            and digest(path) == spec["sha256"], f"Input checksum/size differs: {path}")


def install_inputs(archive, config, root=ROOT):
    """Validate the whole archive and all existing destinations before writing.

    Staging is bounded by the pinned member sizes. Explicit file copies avoid
    tar extraction of links, devices, permissions or path-traversal entries.
    Identical existing files make setup idempotent; conflicts stop setup.
    """
    verify_file(archive, config["archive"])
    files = config["files"]
    destinations = {name: target_path(root, name) for name in files}
    for name, destination in destinations.items():
        if destination.exists():
            verify_file(destination, files[name])
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        require(len(members) == len(files) and {m.name for m in members} == set(files),
                "Archive has missing, duplicate or unexpected members")
        require(all(m.isfile() and m.size == files[m.name]["bytes"] for m in members),
                "Archive member type/size differs; links are not allowed")
        with tempfile.TemporaryDirectory(prefix="tradecraft-input-check-") as tmp:
            staged = {}
            for index, member in enumerate(members):
                path = Path(tmp) / str(index)
                with bundle.extractfile(member) as source, path.open("xb") as out:
                    shutil.copyfileobj(source, out)
                verify_file(path, files[member.name])
                staged[member.name] = path
            for name, source in staged.items():
                destination = target_path(root, name)
                if destination.exists():
                    verify_file(destination, files[name])
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as stream, destination.open("xb") as out:
                    shutil.copyfileobj(stream, out)
    print(f"PASS: {len(files)} pinned input files installed or already identical", flush=True)


def https_url(url):
    p = urlparse(url)
    require(p.scheme == "https" and p.hostname and not p.username and not p.password
            and not p.fragment, "Download requires an HTTPS URL without embedded credentials")
    return url


class HTTPSRedirect(HTTPRedirectHandler):
    """Reject a downgrade BEFORE following a server's redirect."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, config, root=ROOT):
    """Fetch only data, cap transfer size, then verify before accepting it."""
    https_url(url)
    cache = target_path(root, "outputs/reproduction-downloads/" + config["archive"]["name"])
    if cache.exists():
        verify_file(cache, config["archive"])
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading versioned inputs; checksum verification follows", flush=True)
    # Partial downloads are automatically removed; no unverified cache survives.
    with tempfile.TemporaryDirectory(prefix="tradecraft-download-", dir=cache.parent) as tmp:
        path = Path(tmp) / "download.tgz"
        with build_opener(HTTPSRedirect()).open(url, timeout=30) as response, path.open("xb") as out:
            https_url(response.geturl())
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                require(total <= config["archive"]["bytes"], "Download exceeds pinned size")
                out.write(chunk)
        verify_file(path, config["archive"])
        with path.open("rb") as source, cache.open("xb") as out:
            shutil.copyfileobj(source, out)
    return cache


def environment_python():
    return ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def check_python(python, expected):
    version = subprocess.check_output([str(python), "-c", "import platform; print(platform.python_version())"], text=True).strip()
    require(version == expected, f"Need Python {expected}, got {version}; do not bypass the frozen runtime")


def call(python, *args):
    subprocess.run([str(python), *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "check", "run"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Local versioned data-only archive")
    source.add_argument("--url", help="Approved HTTPS location of the identical pinned archive")
    parser.add_argument("--python", type=Path, help="Reuse an already configured interpreter; never install into it")
    parser.add_argument("--output", type=Path, help="New directory for fresh training, required for run")
    args = parser.parse_args()
    config = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(args.action == "setup" or not (args.archive or args.url), "Only setup accepts an archive/URL")
    require((args.action == "run") == (args.output is not None), "Only run requires --output")
    python = args.python.absolute() if args.python else environment_python()
    if args.action == "setup":
        # Catch wrong Python before downloading/extracting or creating a venv.
        check_python(python if args.python else sys.executable, config["python"])
        url = args.url or config.get("download_url")
        if args.archive:
            archive = args.archive
        elif url:
            archive = download(url, config)
        else:
            candidate = ROOT / "outputs" / config["archive"]["name"]
            require(candidate.is_file(), "No download URL is published. Supply --archive PATH from the maintainer.")
            archive = candidate
        install_inputs(archive, config)
        if not args.python:
            if not ENV.exists():
                venv.EnvBuilder(with_pip=True).create(ENV)
            require(python.is_file(), "Existing environment is incomplete; preserve it and inspect manually")
            check_python(python, config["python"])
            call(python, "-m", "pip", "install", "-r", "requirements-reproduce-shared-adam.txt")
        call(python, "-m", "pip", "check")
    check_python(python, config["python"])
    if args.action == "run":
        call(python, "-u", "-m", "src.eval.shared_optimizer_reproduce", "run", "--output", str(args.output.absolute()))
    else:
        call(python, "-m", "src.eval.shared_optimizer_reproduce", "check")
        print("READY: run python scripts/reproduce_shared_adam.py run --output outputs/my-reproduction", flush=True)
        if args.python:
            print(f"Include --python {python} again to reuse your existing environment.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        sys.exit(f"STOP: {error}")
