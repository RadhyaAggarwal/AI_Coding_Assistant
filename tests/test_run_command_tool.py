from tools.run_command import RunCommandTool


def test_runs_command_and_captures_output(tmp_path):
    tool = RunCommandTool(tmp_path, timeout_seconds=10)
    result = tool.run(command="echo hello")
    assert "Exit code: 0" in result
    assert "hello" in result


def test_nonzero_exit_code_reported(tmp_path):
    tool = RunCommandTool(tmp_path, timeout_seconds=10)
    result = tool.run(command='python -c "import sys; sys.exit(3)"')
    assert "Exit code: 3" in result


def test_timeout_is_reported(tmp_path):
    tool = RunCommandTool(tmp_path, timeout_seconds=1)
    result = tool.run(command='python -c "import time; time.sleep(5)"')
    assert "timed out" in result.lower()
