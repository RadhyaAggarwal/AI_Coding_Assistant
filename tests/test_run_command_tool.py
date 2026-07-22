import subprocess
import sys
import time

from tools.run_command import RunCommandTool


def _pid_is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_timeout_kills_the_whole_process_tree_not_just_the_direct_child(tmp_path):
    """Reproduces the real live bug, not just a superficial timeout check:
    a command that spawns its own child process (exactly what Django's
    runserver does via its auto-reloader) must not leave that child
    running after the timeout -- twice observed live to leave the whole
    harness stuck well past the configured timeout, needing a human to
    manually find and kill orphaned processes. subprocess.run(timeout=N)
    on Windows only ever guaranteed killing the immediate tracked child
    (cmd.exe, under shell=True); this proves the real grandchild dies
    too, not just that a timeout message gets returned."""
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    pid_file = tmp_path / "child_pid.txt"

    child_script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    parent_script.write_text(
        "import subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, {str(child_script)!r}])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "child.wait()\n",
        encoding="utf-8",
    )

    tool = RunCommandTool(tmp_path, timeout_seconds=2)
    result = tool.run(f'"{sys.executable}" "{parent_script}"')

    assert "timed out" in result.lower()

    deadline = time.time() + 5
    while not pid_file.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert pid_file.exists(), "child process never started -- test setup issue, not the fix"
    child_pid = int(pid_file.read_text().strip())

    assert not _pid_is_alive(child_pid), (
        "the grandchild process is still running after timeout -- "
        "only the direct child was killed, reproducing the original bug"
    )


def test_runs_command_and_captures_output(tmp_path):
    tool = RunCommandTool(tmp_path, timeout_seconds=10)
    result = tool.run(command="echo hello")
    assert "Exit code: 0" in result
    assert "hello" in result


def test_always_mutates_is_false():
    """Unlike edit_file/create_file, returning normally from run_command
    doesn't guarantee anything actually happened -- a bad path, a typo,
    or a syntax error all just produce an error string, not an
    exception. Live-observed: exactly that kind of no-op "success" used
    to clear every other tool's cached dedup result, letting the model
    re-burn real step budget re-doing lookups it already had answers
    to."""
    assert RunCommandTool.always_mutates is False


def test_dedup_exempt_is_true():
    """A human already confirms every single run_command call, success
    or failure, past or present -- the loop's duplicate-call guard
    should never second-guess that by refusing a repeat before it even
    reaches the confirmation prompt. See tools/base.py's
    Tool.dedup_exempt for the live case this fixes: an explicit request
    to re-run the same test command was silently refused as "already
    called" because it had already succeeded once."""
    assert RunCommandTool.dedup_exempt is True


def test_show_result_is_true():
    """A human should see real command output (e.g. actual pytest
    results), not only the model's own narration of what it showed --
    live-observed: the model's own summary didn't always accurately
    track a test run's real pass/fail count."""
    assert RunCommandTool.show_result is True


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


def test_confirmation_message_is_human_readable_not_a_raw_dict(tmp_path):
    """Live-observed rough edge: the ordinary (non-dangerous) case fell
    back to the generic 'Agent wants to run 'run_command' with arguments
    {...}' message -- a raw Python dict repr shown right below an already
    human-readable progress line saying the same thing. edit_file and
    create_file both got a proper human-facing confirmation message; this
    tool should too."""
    tool = RunCommandTool(tmp_path)
    message = tool.confirmation_message({"command": "pytest tests/test_scratch_grade.py"})
    assert message == "Agent wants to run: pytest tests/test_scratch_grade.py"
    assert "{" not in message


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
