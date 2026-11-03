---
title: Craft@Large Robot Studio
emoji: 🧵
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
short_description: Workshop groups edit and version Reachy Mini's personality
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Craft@Large Robot Studio

A web page where Craft@Large groups write their robot's one-liners, pick a voice and abilities, save versions, and try them on a Reachy Mini. Built on Pollen Robotics' Reachy Mini conversation app.

## Set up (once per laptop)

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). On a Mac with Homebrew:

```bash
brew install python@3.12 uv
```

Then, from this folder:

```bash
uv venv --python python3.12 .venv
source .venv/bin/activate
uv sync --frozen
```

Keep this folder in your home folder, not on an iCloud-synced Desktop. iCloud hides files the environment needs.

## Run with the robot

1. Turn on the robot and put it on the same Wi-Fi as the laptop.
2. Find the robot's IP address in the Reachy Mini Control app (Make sure this app is open).
4. Start the app, using your robot's IP address:

```bash
cd ~/craft_at_large_app
source .venv/bin/activate
craft-at-large-app --ui --robot-host 192.168.1.42
```

5. Open http://localhost:7860/workshop and pick a group.

The app starts the robot by itself. Stop the app with Ctrl+C.

## Run with the simulator (no robot)

In one terminal, start the simulated robot:

```bash
source .venv/bin/activate
mjpython -m reachy_mini.daemon.app.main --sim    # Mac
reachy-mini-daemon --sim                          # Linux
```

In a second terminal, start the app:

```bash
source .venv/bin/activate
craft-at-large-app --ui
```

Then open http://localhost:7860/workshop. Close the simulator before using a real robot.

## Preview the page only (no robot, no simulator)

```bash
source .venv/bin/activate
python preview_workshop.py
```

Then open http://localhost:7861/workshop. "Try on robot" only pretends.

## Options

| Option | What it does |
|---|---|
| `--robot-host <IP>` | Connect to the robot at this IP address |
| `--record` | Start recording when the app starts (or use "Start recording" at the bottom of the page) |
| `--no-camera` | Turn the camera off, including recording photos |
| `CRAFT_PHOTO_SECONDS=60` | Seconds between recording photos (default 300, `0` for none) |
| `CRAFT_SILENCE_SECONDS=20` | Seconds of quiet before the robot is told the room is quiet (default 30, `0` for off) |

Put the settings in front of the command, for example `CRAFT_PHOTO_SECONDS=60 craft-at-large-app --ui --robot-host 192.168.1.42`.

## Where things are saved

- `saved_work/` holds each group's versions.
- `recordings/` holds transcripts and photos. Clicking "Stop recording" also downloads the session as a zip to the laptop. When the app runs on the robot, the robot deletes its copy once the zip has fully downloaded.
- `profiles/craft_group_*/` holds each group's Default prompt (`profile.md`), name (`name.txt`) and card photo (`photo.png`).

`saved_work/` and `recordings/` contain participant data. They stay on the laptop and are not uploaded to GitHub.
