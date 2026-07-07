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


def test_confirmation_message_generic_for_ordinary_command(tmp_path):
    tool = RunCommandTool(tmp_path)
    message = tool.confirmation_message({"command": "pytest tests/"})
    assert "WARNING" not in message


def test_confirmation_message_warns_for_git_push():
    """Reproduces the exact live safety finding: a bigger model, asked
    only to fix a bug and run tests, went on to unprompted run git push
    -- the confirmation prompt must make that impossible to miss, not
    look like every other routine command."""
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": "git push origin main"})
    assert "WARNING" in message
    assert "remote" in message


def test_confirmation_message_warns_for_git_commit():
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": 'git commit -m "wip"'})
    assert "WARNING" in message


def test_confirmation_message_warns_for_git_reset_hard():
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": "git reset --hard HEAD~1"})
    assert "WARNING" in message
    assert "irreversibly" in message


def test_confirmation_message_warns_for_rm_rf():
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": "rm -rf build/"})
    assert "WARNING" in message


def test_confirmation_message_warns_for_rm_rf_flag_order_variant():
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": "rm -fr build/"})
    assert "WARNING" in message


def test_confirmation_message_warns_for_powershell_recurse_force_delete():
    tool = RunCommandTool(".")
    message = tool.confirmation_message({"command": "Remove-Item -Recurse -Force build"})
    assert "WARNING" in message


def test_confirmation_message_does_not_flag_unrelated_git_commands():
    """git status/diff/log are safe, read-only operations -- they must
    not be flagged, or the warning stops meaning anything."""
    tool = RunCommandTool(".")
    for command in ["git status", "git diff", "git log --oneline -5"]:
        message = tool.confirmation_message({"command": command})
        assert "WARNING" not in message, f"false positive for: {command}"
