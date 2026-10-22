"""Portable bootstrap safety tests; synthetic archives, no downloads/training."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import reproduce_shared_adam as setup


def bundle(root, items):
    archive = root / "input.tgz"
    files = {}
    with tarfile.open(archive, "w:gz") as tar:
        for name, payload, kind in items:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.type = kind
            if kind == tarfile.SYMTYPE:
                member.linkname = "elsewhere"
            tar.addfile(member, io.BytesIO(payload))
            files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    config = {"archive": {"bytes": archive.stat().st_size, "sha256": setup.digest(archive)}, "files": files}
    return archive, config


class BootstrapTests(unittest.TestCase):
    def test_mode_dispatch_is_explicit_and_strict_by_default(self):
        for mode, module in ((None, "reproduce"), ("strict", "reproduce"), ("portable", "portable")):
            for action in ("check", "run"):
                args = ["reproduce_shared_adam.py", action, "--python", "my-python"]
                if mode:
                    args += ["--mode", mode]
                if action == "run":
                    args += ["--output", "outputs/new"]
                with patch.object(setup.sys, "argv", args), patch.object(setup, "check_python"), \
                     patch.object(setup, "call") as call:
                    setup.main()
                    self.assertIn("src.eval.shared_optimizer_" + module, call.call_args.args)
                    self.assertIn(action, call.call_args.args)

    def test_install_and_idempotent_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, config = bundle(root, [("outputs/data/a", b"hello", tarfile.REGTYPE)])
            setup.install_inputs(archive, config, root)
            setup.install_inputs(archive, config, root)
            self.assertEqual((root / "outputs/data/a").read_bytes(), b"hello")

    def test_bad_archive_digest_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, config = bundle(root, [("outputs/data/a", b"hello", tarfile.REGTYPE)])
            config["archive"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                setup.install_inputs(archive, config, root)
            self.assertFalse((root / "outputs").exists())

    def test_existing_conflict_preserves_all_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, config = bundle(root, [("outputs/data/a", b"new", tarfile.REGTYPE),
                                           ("outputs/data/b", b"expected", tarfile.REGTYPE)])
            conflict = root / "outputs/data/b"
            conflict.parent.mkdir(parents=True)
            conflict.write_bytes(b"user work")
            with self.assertRaises(ValueError):
                setup.install_inputs(archive, config, root)
            self.assertFalse((conflict.parent / "a").exists())
            self.assertEqual(conflict.read_bytes(), b"user work")

    def test_per_file_corruption_is_detected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, config = bundle(root, [("outputs/data/a", b"hello", tarfile.REGTYPE)])
            config["files"]["outputs/data/a"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                setup.install_inputs(archive, config, root)
            self.assertFalse((root / "outputs").exists())

    def test_paths_outside_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("../bad", "/tmp/bad", "outputs/../bad", "src/evil.py",
                         "outputs/a\\b", "outputs/C:/bad", "outputs//a", "outputs/./a"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    setup.target_path(tmp, name)

    def test_duplicates_and_symlinks_are_rejected(self):
        for items in ([("outputs/a", b"", tarfile.SYMTYPE)],
                      [("outputs/a", b"x", tarfile.REGTYPE)] * 2):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive, config = bundle(root, items)
                with self.assertRaises(ValueError):
                    setup.install_inputs(archive, config, root)
                self.assertFalse((root / "outputs").exists())

    def test_existing_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outside").mkdir()
            try:
                (root / "outputs").symlink_to(root / "outside", target_is_directory=True)
            except OSError:
                self.skipTest("Host does not allow unprivileged symlinks")
            with self.assertRaisesRegex(ValueError, "Symlink"):
                setup.target_path(root, "outputs/a")

    def test_download_rejects_plain_http_and_credentials(self):
        for url in ("http://example.com/a", "file:///tmp/a", "https://u:p@example.com/a"):
            with self.assertRaises(ValueError):
                setup.https_url(url)
        self.assertEqual(setup.https_url("https://example.com/a"), "https://example.com/a")

    def test_runtime_mismatch_has_clear_error(self):
        with patch.object(setup.subprocess, "check_output", return_value="3.12.1\n"):
            with self.assertRaisesRegex(ValueError, "Need Python 3.13.9"):
                setup.check_python("python", "3.13.9")

    def test_https_redirect_rejects_downgrade(self):
        with self.assertRaises(ValueError):
            setup.HTTPSRedirect().redirect_request(None, None, 302, "", {}, "http://example.com/a")

    def test_download_checks_then_reuses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = b"synthetic archive"
            config = {"archive": {"name": "a.tgz", "bytes": len(payload),
                                  "sha256": hashlib.sha256(payload).hexdigest()}}
            response = io.BytesIO(payload)
            response.geturl = lambda: "https://example.com/asset"
            with patch.object(setup, "build_opener") as opener:
                opener.return_value.open.return_value = response
                path = setup.download("https://example.com/asset", config, tmp)
                self.assertEqual(path.read_bytes(), payload)
                self.assertEqual(setup.download("https://example.com/asset", config, tmp), path)
                self.assertEqual(opener.return_value.open.call_count, 1)

    def test_oversized_download_is_not_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = io.BytesIO(b"too long")
            response.geturl = lambda: "https://example.com/asset"
            config = {"archive": {"name": "a.tgz", "bytes": 1, "sha256": "0" * 64}}
            with patch.object(setup, "build_opener") as opener:
                opener.return_value.open.return_value = response
                with self.assertRaisesRegex(ValueError, "exceeds pinned size"):
                    setup.download("https://example.com/asset", config, tmp)
            cache = Path(tmp) / "outputs/reproduction-downloads"
            self.assertEqual(list(cache.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
