"""Verify or reconstruct retired experiments without changing the active tree.

Frozen receipts contain historical filenames and source hashes. Relocating the
bytes is safe; rewriting those receipts to pretend they ran from a new path is
not. A disposable reconstruction restores the old layout for regression tests.
Only Git-tracked active files and the checked archive are copied: private data,
checkpoints, virtual environments and unrelated untracked drafts stay excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def relative_path(value):
    """Reject traversal and Windows-style absolute paths on every platform."""
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a portable relative path')
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        raise ValueError('Unsafe or noncanonical archive path: ' + value)
    return path


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'archive/manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or not manifest.get('files'):
        raise ValueError('Unsupported or empty archive manifest')
    originals, archived = set(), set()
    for entry in manifest['files']:
        original = relative_path(entry['original'])
        saved = relative_path(entry['archived'])
        group = entry['group']
        if group not in {'classifier_search', 'early_runs', 'proposals'}:
            raise ValueError('Unknown archive group')
        if saved != PurePosixPath('archive', group) / original:
            raise ValueError('Archive mapping changed')
        if original.parts[0] not in {'src', 'scripts', 'tests', 'reports', 'research_notes'}:
            raise ValueError('Archive source outside approved directories')
        if str(original) in originals or str(saved) in archived:
            raise ValueError('Duplicate archive mapping')
        originals.add(str(original))
        archived.add(str(saved))
        source = root / saved
        if source.is_symlink() or source.resolve().is_relative_to(root) is False:
            raise ValueError('Archive path must be a local regular file')
        data = source.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Archived bytes changed: ' + str(saved))
    return manifest


def tracked_files(root):
    """Use the Git index, never a recursive walk into local model/data folders."""
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=root, check=True,
                            capture_output=True)
    return [p.decode('utf-8') for p in result.stdout.split(b'\0') if p]


def restore(destination, root=ROOT):
    root = Path(root).resolve()
    manifest = verify(root)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve existing destination: ' + str(destination))
    # Validate every planned copy before creating anything. Missing tracked paths
    # are permitted only for the archive moves before their first commit.
    old_paths = {entry['original'] for entry in manifest['files']}
    copies = []
    for name in tracked_files(root):
        path = relative_path(name)
        if path.parts[0] == 'archive' or name in old_paths:
            continue
        source = root / path
        if not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Active tracked file missing or unsafe: ' + name)
        copies.append((source, destination / path))
    for entry in manifest['files']:
        copies.append((root / entry['archived'], destination / entry['original']))
    destination.mkdir(parents=True, exist_ok=False)
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return manifest


def archive_tests(manifest):
    return sorted(entry['original'][:-3].replace('/', '.')
                  for entry in manifest['files']
                  if entry['original'].startswith('tests/test_')
                  and entry['original'].endswith('.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'restore', 'test'], nargs='?', default='verify')
    parser.add_argument('--output', type=Path, help='New directory for an explicit historical reconstruction')
    args = parser.parse_args()
    if args.action == 'restore':
        if args.output is None:
            parser.error('restore requires --output')
        manifest = restore(args.output)
        print(f'Restored {len(manifest["files"])} archived files into {args.output}; no experiment launched')
    else:
        if args.output is not None:
            parser.error('--output is only supported with restore')
        manifest = verify()
        print(f'PASS: {len(manifest["files"])} archived files are byte-identical', flush=True)
        if args.action == 'test':
            # Only synthetic/receipt regressions run. No downloaded datasets or
            # checkpoint artifacts are copied into this disposable view.
            with tempfile.TemporaryDirectory(prefix='tradecraft-archive-') as temp:
                view = Path(temp) / 'historical'
                restore(view)
                result = subprocess.run([sys.executable, '-m', 'unittest', '-q',
                                         *archive_tests(manifest)], cwd=view)
                raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
