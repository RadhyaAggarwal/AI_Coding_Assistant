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

- `ollama pull <model>` — downloads a model (e.g. `qwen2.5-coder:7b`).
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
- [Docker](https://docs.docker.com/get-docker/) installed. On Linux
  (recommended for a machine meant to stay running as a server):
  ```
  curl -fsSL https://get.docker.com | sh
  ```
  On Windows, install Docker Desktop instead (requires WSL2).
- If using an NVIDIA GPU, also install the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  — without it, Docker can't give containers access to the GPU even
  though it's physically present. Not needed for a CPU-only server; see
  the note in `deploy/docker-compose.yml` for what to remove if so.

### Start Ollama

From this project's root on the server machine:

```
docker compose -f deploy/docker-compose.yml up -d
```

This pulls the official `ollama/ollama` image, starts it in the
background, and configures it to restart automatically if it crashes or
fails its health check (see the comments in that file for exactly what
each setting does and why).

### Pull the model

Same model names this project already uses elsewhere — run this once,
on the server:

```
docker exec ollama ollama pull qwen2.5-coder:7b
```

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
     name: "qwen2.5-coder:7b"   # must match what was pulled on the server
   ```
4. To work on a specific project, `cd` into *that project's own folder*
   first, then run this agent's `main.py` from there (using its full
   path, or add it to your `PATH`). `config.yaml`'s `project.root_path:
   "."` resolves relative to whatever directory you're standing in when
   you run it — confirmed by testing this directly — so the agent
   operates on whichever project you're currently in, not on wherever
   this project's own code happens to be installed.

## 3. Operating the server

- **Logs**: `docker compose -f deploy/docker-compose.yml logs -f`
- **Manual restart**: `docker compose -f deploy/docker-compose.yml restart`
- **Stop entirely**: `docker compose -f deploy/docker-compose.yml down`
  (the downloaded model is preserved in the `ollama_data` volume even
  after this — `up -d` again won't need to re-pull it)
- **Pull/change models**: `docker exec ollama ollama pull <model>`, then
  update every client's `config.yaml` to match.
- **Concurrency**: Ollama can serve multiple requests at once
  (`OLLAMA_NUM_PARALLEL`), bounded by the server's actual hardware —
  several people can genuinely get simultaneous responses, but enough
  simultaneous heavy requests will queue rather than instantly complete.
  This is normal for any shared compute resource.
- **The healthcheck in `deploy/docker-compose.yml` is a mitigation, not
  a confirmed fix**, for a hang/timeout issue this project has hit
  before and not fully root-caused (see project memory /
  conversation history for the investigation). It should make an
  unresponsive Ollama recover automatically within roughly a minute
  instead of requiring a human to notice and intervene — but if it
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
