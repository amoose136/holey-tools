"""Runner selection and process failure semantics, without requiring Blender."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import run


class RunnerTests(unittest.TestCase):
    def test_default_discovery_excludes_parked_tools(self):
        active = {test.path.name for test in run.select_tests()}
        self.assertIn('test_modes.py', active)
        self.assertIn('test_runner.py', active)
        self.assertNotIn('test_extract_axis.py', active)
        all_tests = {test.path.name for test in run.select_tests(include_experimental=True)}
        self.assertEqual(all_tests - active, run.EXPERIMENTAL_TESTS)

    def test_aliases_are_deduplicated_and_explicit_experimental_is_allowed(self):
        tests = run.select_tests(['modes', 'test_modes', 'test_modes.py', 'extract_axis'])
        self.assertEqual([test.path.name for test in tests], ['test_modes.py', 'test_extract_axis.py'])
        with self.assertRaisesRegex(ValueError, 'Unknown test'):
            run.select_tests(['../not_a_test'])

    def test_command_always_isolates_native_undo(self):
        test = run.select_tests(['undo_lifecycle'])[0]
        command = run.command_for(test, '/path with spaces/blender')
        self.assertEqual(command, ['/path with spaces/blender', '--background', '--factory-startup',
                                   '--python-exit-code', '1', '--python', str(test.path)])
        self.assertEqual(command.count('--python'), 1)
        with self.assertRaises(ValueError):
            run.command_for(test)
        python = run.select_tests(['resize_state'])[0]
        self.assertFalse(python.needs_blender)
        self.assertEqual(run.command_for(python), [sys.executable, str(python.path)])

    def test_explicit_and_environment_executable_resolution(self):
        with patch.dict(os.environ, {'BLENDER': '/env/blender'}), patch.object(
            run.shutil, 'which', side_effect=lambda name: name if name in ('/env/blender', '/explicit/blender') else None
        ):
            self.assertEqual(run.find_blender('/explicit/blender'), '/explicit/blender')
            self.assertEqual(run.find_blender(), '/env/blender')
            with self.assertRaisesRegex(ValueError, 'not found'):
                run.find_blender('/missing/blender')

    def test_missing_executable_reports_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(run.shutil, 'which', return_value=None), patch.object(
            Path, 'is_file', return_value=False
        ):
            with self.assertRaisesRegex(ValueError, '--blender PATH'):
                run.find_blender()

    def test_crashes_and_output_are_not_reported_as_passes(self):
        script = run.select_tests(['modes'])[0]
        completed = subprocess.CompletedProcess([], -11, 'native output\n', 'crash\n')
        with patch.object(run.subprocess, 'run', return_value=completed) as child:
            result = run.run_test(script, '/blender', timeout=23)
        self.assertEqual(result.returncode, -11)
        self.assertEqual(result.output, 'native output\ncrash\n')
        self.assertEqual(child.call_args.kwargs, dict(cwd=run.ROOT, capture_output=True, text=True, timeout=23))
        self.assertGreaterEqual(result.seconds, 0)

    def test_timeouts_keep_partial_output_and_fail(self):
        error = subprocess.TimeoutExpired(['blender'], 5, output=b'partial\xff', stderr=b'error\n')
        with patch.object(run.subprocess, 'run', side_effect=error):
            result = run.run_test(run.select_tests(['modes'])[0], '/blender', timeout=5)
        self.assertEqual(result.returncode, 124)
        self.assertIn('partial', result.output)
        self.assertIn('error', result.output)
        self.assertIn('Timed out after 5s', result.output)

    def test_failed_launch_is_not_a_pass(self):
        with patch.object(run.subprocess, 'run', side_effect=OSError('launch failed')):
            result = run.run_test(run.select_tests(['modes'])[0], '/blender')
        self.assertEqual(result.returncode, 127)
        self.assertIn('launch failed', result.output)

    def test_list_and_python_only_need_no_blender_installation(self):
        with redirect_stdout(io.StringIO()), patch.object(run, 'find_blender') as find:
            self.assertEqual(run.main(['--list']), 0)
            with patch.object(run, 'run_test', side_effect=lambda script, *_args: run.Result(script, 0, '', 0)):
                self.assertEqual(run.main(['resize_state']), 0)
        find.assert_not_called()

    def test_parallel_suite_finishes_other_tests_and_reports_failure(self):
        def result(script, *_args):
            return run.Result(script, 1 if script.path.stem == 'test_modes' else 0, 'diagnostic\n', .1)
        output = io.StringIO()
        with redirect_stdout(output), patch.object(run, 'find_blender', return_value='/blender'), patch.object(
            run, 'run_test', side_effect=result
        ) as child:
            self.assertEqual(run.main(['modes', 'undo_lifecycle', '--jobs', '2']), 1)
        self.assertEqual(child.call_count, 2)
        self.assertIn('diagnostic', output.getvalue())
        self.assertIn('1/2 scripts passed; 1 failed.', output.getvalue())

    def test_invalid_limits_and_test_names_fail_before_launch(self):
        for args in (['--jobs', '0'], ['--timeout', '-1'], ['missing_test']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run.main(args)
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
