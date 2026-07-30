# Deploying a shared Ollama server

This is for running one shared model server that everyone on the same
local network points this project's agent at, instead of each person
installing Ollama and downloading a multi-gigabyte model individually.

If you just want to run everything on your own single machine, you don't
need any of this — see the Quickstart in `README.md` instead.

## What Ollama actually is

[Ollama](https://ollama.com) is the tool this whole project is built on
top of: it downloads and runs open-source language models (like
`qwen2.5-coder`) on a machine, and exposes them over a local HTTP API
(`/api/chat`, `/api/tags`, etc. — port `11434` by default). This
project's `model_interface/ollama_adapter.py` is the only file that
talks to that API; nothing else in the codebase knows or cares whether
Ollama is running on the same machine or a different one on the network.

The handful of commands worth knowing:

- `ollama pull <model>` — downloads a model (e.g. `qwen2.5-coder:14b`,
  the size currently configured in this project's `config.yaml` — see
  the note on model size in the Requirements section below).
  Only needs to happen once per machine that will actually run the model.
- `ollama list` — shows which models are downloaded.
- `ollama run <model>` — an interactive chat session directly in the
  terminal, mostly useful for manually poking at a model; this project
  doesn't use this command, it talks to `ollama serve`'s API instead.
- `ollama serve` — starts the background server that exposes the HTTP
  API. This is the piece that needs to be running for anything in this
  project to work, whether it's started directly or (as recommended
  below) inside a container.

## Architecture recap

One machine (the "server") runs Ollama and holds the model. Every other
team member's machine runs a normal copy of *this project's code* (no
Ollama, no model download) with `config.yaml` pointed at the server's
address. Only the model itself lives on the server — every tool call
(`read_file`, `edit_file`, `run_command`, etc.) still runs entirely on
each user's own machine, against their own files. The server never sees
or touches anyone's project files.

## 1. Set up the server

### Requirements

- A machine reachable by everyone on the local network. A GPU is
  strongly recommended — this project's own CPU-only testing has run
  roughly 1-4 minutes per model call, which multiplies badly once
  several people are sharing one machine.
- **Model size, chosen deliberately, not the smallest option:** this
  project currently defaults to `qwen2.5-coder:14b` (`config.yaml`'s
  `model.name`), not the smaller `qwen2.5-coder:7b`. Concrete reason:
  given a bug report describing a symptom in one function, but whose
  actual root cause lived in a different function it called, the 7B
  model never investigated the dependency at all and invented a wrong
  fix instead; the 14B model correctly traced it and fixed the real
  cause. This is a real hardware tradeoff, not free: 14B needs more
  RAM/VRAM than 7B, which matters more here than on a single personal
  machine since one shared server absorbs everyone's requests.
  Drop to `qwen2.5-coder:7b` (pull it instead, and update every client's
  `config.yaml` to match) if the server hardware can't comfortably run
  14B, especially once several people are sharing it concurrently.
- **Linux or Windows server:** [Docker](https://docs.docker.com/get-docker/)
  installed. On Linux (recommended for a machine meant to stay running
  as a server):
  ```
  curl -fsSL https://get.docker.com | sh
  ```
  On Windows, install Docker Desktop instead (requires WSL2). If using
  an NVIDIA GPU, also install the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  — without it, Docker can't give containers access to the GPU even
  though it's physically present. Not needed for a CPU-only server; see
  the note in `deploy/docker-compose.yml` for what to remove if so.
- **Mac server: skip Docker, run Ollama natively instead.** Docker
  Desktop on Mac can't pass Apple Silicon's GPU through to containers,
  so Ollama running inside Docker there would be CPU-only — slower
  than running it directly, which does get GPU acceleration. See
  "Start Ollama on macOS" below instead of the Docker steps.

### Start Ollama on Linux/Windows (Docker)

From this project's root on the server machine:

```
docker compose -f deploy/docker-compose.yml up -d
```

This pulls the official `ollama/ollama` image, starts it in the
background, and configures it to restart automatically if it crashes or
fails its health check (see the comments in that file for exactly what
each setting does and why).

### Start Ollama on macOS (no Docker)

Install Ollama via the `.dmg` from https://ollama.com/download, or
`brew install ollama` — same as a single-user setup (see `README.md`).

**Keep it running automatically, including across a reboot:**

1. Find where Ollama is actually installed:
   ```
   which ollama
   ```
2. Create a file at `~/Library/LaunchAgents/com.ollama.serve.plist`
   with this content, replacing `/path/to/ollama` with the real path
   from step 1:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.ollama.serve</string>
       <key>ProgramArguments</key>
       <array>
           <string>/path/to/ollama</string>
           <string>serve</string>
       </array>
       <key>RunAtLoad</key>
       <true/>
       <key>KeepAlive</key>
       <true/>
       <key>StandardOutPath</key>
       <string>/tmp/ollama.log</string>
       <key>StandardErrorPath</key>
       <string>/tmp/ollama.log</string>
   </dict>
   </plist>
   ```
3. Activate it:
   ```
   launchctl load ~/Library/LaunchAgents/com.ollama.serve.plist
   ```

`RunAtLoad` starts Ollama the next time this Mac boots or this user
logs in; `KeepAlive` restarts it automatically if the process ever
crashes or is quit. Together these do the same job as Docker's
`restart: unless-stopped` on Linux/Windows.

**Optional, more advanced — also recover from a hang, not just a
crash.** The setup above only restarts Ollama if it actually exits; it
won't notice Ollama being stuck-but-still-running (a hang problem this
project has hit before — see "Operating the server" below). To also
catch that, save this as `~/ollama_watchdog.sh`:

```bash
#!/bin/bash
if ! curl -s -o /dev/null --max-time 10 http://localhost:11434/api/tags; then
    pkill -f "ollama serve"
fi
```

(This is a minimal version — it restarts on the very first failed
check, rather than waiting for a few in a row first. Fine as a starting
point; a more careful version would avoid reacting to one slow-but-fine
request.) Make it executable and run it every 2 minutes via `cron`:

```
chmod +x ~/ollama_watchdog.sh
crontab -e
```

then add this line inside the editor that opens:

```
*/2 * * * * /Users/<your-username>/ollama_watchdog.sh
```

Killing the process is enough on its own — `launchd`'s `KeepAlive` from
above will restart it automatically right after.

### Keep the machine itself awake

A "server" here is just a regular computer — nothing stops it from
going to sleep like any other one, and a sleeping machine stops
responding to the network entirely until it wakes up (indistinguishable
from being down, from a client's point of view). If this machine needs
to stay available continuously, turn off sleep in its power settings:
macOS, System Settings → Battery/Energy; Windows, Settings → Power &
Sleep. This is a separate concern from crash/hang recovery above —
sleep is about the *machine* being reachable at all, while `launchd`
(or Docker's restart policy) is about the Ollama *process* recovering
once the machine already is reachable.

(A full shutdown is different from sleep — everything stops, and
starting back up is exactly what `RunAtLoad`/Docker's restart policy
already handle automatically, same as above.)

### Pull the model

Same model name this project's `config.yaml` already uses — run this
once, on the server (drop the `docker exec ollama` prefix if running
Ollama natively on macOS, per above):

```
docker exec ollama ollama pull qwen2.5-coder:14b
```

If anyone on the team wants `semantic_search` (meaning-based code
search — see `README.md`), also pull a second, much smaller embedding
model on the same server the same way:

```
docker exec ollama ollama pull nomic-embed-text
```

No separate deployment for this — it's just another model served by
the same Ollama instance on the same port. Each client that wants it
enabled sets `model.embedding_name: "nomic-embed-text"` in their own
`config.yaml` alongside `endpoint_url`, same as below.

### Verify it's reachable from the network

From a *different* machine on the same LAN:

```
curl http://<server-lan-ip>:11434/api/tags
```

You should get back JSON listing the model you just pulled. Find
`<server-lan-ip>` however you normally would on your network (e.g.
`ipconfig` on the server itself, on Windows).

**One thing to check here that we haven't verified yet:** earlier
testing against a Cloudflare tunnel found Ollama rejecting requests
whose `Host` header wasn't `localhost`/`127.0.0.1` (worked around at the
time with a small reverse proxy). We've never actually tested whether a
plain LAN IP triggers the same rejection — it might not, since a LAN IP
looks a lot more like a normal request than a random public tunnel
hostname. If the `curl` command above fails with something like `403
Forbidden` rather than a connection error, that's what's happening, and
the fix is the same kind of small reverse proxy used for the Colab
setup (`OLLAMA_ORIGINS` alone was confirmed *not* to fix this — don't
waste time on that route first).

## 2. Set up each client machine

Each team member's own machine, for working on their own projects:

1. Get a copy of this project's code (clone/copy it — no Ollama install,
   no model download needed here at all).
2. Install Python dependencies: `pip install -r requirements.txt`.
3. Edit `config.yaml`:
   ```yaml
   model:
     endpoint_url: "http://<server-lan-ip>:11434"
     name: "qwen2.5-coder:14b"   # must match what was pulled on the server
     # embedding_name: "nomic-embed-text"   # optional, only if you want semantic_search
   ```
4. To work on a specific project, `cd` into *that project's own folder*
   first, then run this agent's `main.py` from there (using its full
   path, or add it to your `PATH`). `config.yaml`'s `project.root_path:
   "."` resolves relative to whatever directory you're standing in when
   you run it — confirmed by testing this directly — so the agent
   operates on whichever project you're currently in, not on wherever
   this project's own code happens to be installed.

## 3. Operating the server

**On Linux/Windows (Docker):**

- **Logs**: `docker compose -f deploy/docker-compose.yml logs -f`
- **Manual restart**: `docker compose -f deploy/docker-compose.yml restart`
- **Stop entirely**: `docker compose -f deploy/docker-compose.yml down`
  (the downloaded model is preserved in the `ollama_data` volume even
  after this — `up -d` again won't need to re-pull it)
- **Pull/change models**: `docker exec ollama ollama pull <model>`, then
  update every client's `config.yaml` to match.

**On macOS (native, via `launchd`):**

- **Logs**: `tail -f /tmp/ollama.log` (or wherever `StandardOutPath` was
  set to, in the plist from the setup step above)
- **Manual restart**: `launchctl unload ~/Library/LaunchAgents/com.ollama.serve.plist && launchctl load ~/Library/LaunchAgents/com.ollama.serve.plist`
- **Stop entirely**: `launchctl unload ~/Library/LaunchAgents/com.ollama.serve.plist`
  (the downloaded model stays on disk regardless — nothing needs
  re-pulling when you load it again)
- **Pull/change models**: `ollama pull <model>`, then update every
  client's `config.yaml` to match.

**Both platforms:**

- **Concurrency**: Ollama can serve multiple requests at once
  (`OLLAMA_NUM_PARALLEL`), bounded by the server's actual hardware —
  several people can genuinely get simultaneous responses, but enough
  simultaneous heavy requests will queue rather than instantly complete.
  This is normal for any shared compute resource.
- **The healthcheck in `deploy/docker-compose.yml` (or the equivalent
  watchdog script on a macOS server, see above) is a mitigation, not
  a confirmed fix**, for a hang/timeout issue this project has hit
  before and not fully root-caused — see commit `30108b8` ("Add a
  fail-fast health check before each model request") for the original
  investigation and what was actually confirmed vs. still uncertain.
  It should make an unresponsive Ollama recover automatically within
  roughly a minute instead of requiring a human to notice and
  intervene — but if it
  turns out to recur even with this in place, that's a real finding
  worth investigating further, not something to assume is already
  solved.

## Security notes

- This setup assumes the server is only reachable on a trusted local
  network, not the public internet. There's no authentication in front
  of Ollama here — anyone who can reach the server's IP can use it.
- Sharing the model server does **not** mean sharing file/execution
  access. Every tool with real side effects (`edit_file`, `create_file`,
  `run_command`) still requires human confirmation on the *client's own
  machine*, against the client's own files — the server only ever sees
  conversation text, never anyone's filesystem.
