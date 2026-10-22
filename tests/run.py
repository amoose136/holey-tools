#!/usr/bin/env python3
"""Run every script in its own Python or factory-startup Blender process."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
PYTHON_TESTS = {'test_resize_dimensions.py', 'test_resize_state.py', 'test_runner.py'}
EXPERIMENTAL_TESTS = {'test_extract_axis.py'}


@dataclass(frozen=True)
class TestScript:
    path: Path

    @property
    def needs_blender(self):
        return self.path.name not in PYTHON_TESTS


@dataclass(frozen=True)
class Result:
    script: TestScript
    returncode: int
    output: str
    seconds: float


def select_tests(names=(), *, include_experimental=False):
    """Select known scripts, accepting modes, test_modes, or test_modes.py."""
    known = {path.name: TestScript(path) for path in sorted(TESTS_DIR.glob('test_*.py'))}
    if not names:
        return [test for name, test in known.items()
                if include_experimental or name not in EXPERIMENTAL_TESTS]
    selected = []
    for name in names:
        name = name.removesuffix('.py')
        name = name if name.startswith('test_') else f'test_{name}'
        try:
            script = known[f'{name}.py']
        except KeyError:
            raise ValueError(f'Unknown test: {name}. Use --list to see available tests.') from None
        if script not in selected:
            selected.append(script)
    return selected


def find_blender(explicit=None):
    candidate = explicit or os.environ.get('BLENDER') or shutil.which('blender')
    if candidate:
        resolved = shutil.which(str(Path(candidate).expanduser()))
        if resolved:
            return resolved
        raise ValueError(f'Blender executable not found or not executable: {candidate}')
    mac_app = Path('/Applications/Blender.app/Contents/MacOS/blender')
    if mac_app.is_file() and os.access(mac_app, os.X_OK):
        return str(mac_app)
    raise ValueError('Blender not found. Pass --blender PATH, set BLENDER, or add it to PATH.')


def command_for(script, blender=None):
    if not script.needs_blender:
        return [sys.executable, str(script.path)]
    if blender is None:
        raise ValueError('A Blender executable is required for mesh/operator tests.')
    return [blender, '--background', '--factory-startup', '--python-exit-code', '1',
            '--python', str(script.path)]


def run_test(script, blender=None, timeout=180):
    start = time.monotonic()
    try:
        completed = subprocess.run(command_for(script, blender), cwd=ROOT,
                                   capture_output=True, text=True, timeout=timeout)
        code, output = completed.returncode, completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        code = 124
        output = decoded(error.stdout) + decoded(error.stderr) + f'\nTimed out after {timeout}s.\n'
    except OSError as error:
        code, output = 127, f'{error}\n'
    return Result(script, code, output, time.monotonic() - start)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tests', nargs='*', help='Test names; defaults to all active tests')
    parser.add_argument('--list', action='store_true', help='List selected tests without running them')
    parser.add_argument('--blender', help='Blender executable (otherwise BLENDER/PATH/standard macOS app)')
    parser.add_argument('--include-experimental', action='store_true', help='Include parked Extract Axis tests')
    parser.add_argument('--jobs', type=positive_int, default=1, help='Independent processes to run concurrently')
    parser.add_argument('--timeout', type=positive_int, default=180, help='Timeout per script in seconds')
    parser.add_argument('--verbose', action='store_true', help='Print successful test output too')
    args = parser.parse_args(argv)
    try:
        scripts = select_tests(args.tests, include_experimental=args.include_experimental)
        if args.list:
            for script in scripts:
                print(f'{script.path.stem} ({"Blender" if script.needs_blender else "Python"})')
            return 0
        blender = find_blender(args.blender) if any(test.needs_blender for test in scripts) else None
    except ValueError as error:
        parser.error(str(error))

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = []
        for script in scripts:
            print(f'QUEUED {script.path.stem}', flush=True)
            pending.append(pool.submit(run_test, script, blender, args.timeout))
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            label = 'PASS' if result.returncode == 0 else f'FAIL ({result.returncode})'
            print(f'{label} {result.script.path.stem} ({result.seconds:.2f}s)', flush=True)
            if result.returncode != 0 or args.verbose:
                print(result.output, end='' if result.output.endswith('\n') else '\n', flush=True)
    failures = sum(result.returncode != 0 for result in results)
    print(f'{len(results) - failures}/{len(results)} scripts passed; {failures} failed.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
