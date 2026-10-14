"""Who is speaking, from the call's audio (voice/listen.py)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")
from voice import listen  # noqa: E402


def test_the_detector_needs_sound_to_start_and_quiet_to_stop():
    v = listen.Vad()
    assert [v.feed(100) for _ in range(5)] == [None] * 5  # the line's own noise
    assert v.feed(3000) is None  # one loud window is a click
    assert v.feed(3000) == "started"  # two is a voice
    assert [v.feed(3000) for _ in range(10)] == [None] * 10
    assert [v.feed(100) for _ in range(listen.STOP_WINDOWS - 1)] == [None] * (listen.STOP_WINDOWS - 1)  # a breath
    assert v.feed(3000) is None and v.on  # and on they go
    assert [v.feed(100) for _ in range(listen.STOP_WINDOWS)][-1] == "stopped"


def test_a_noisy_line_is_not_a_voice():
    v = listen.Vad()
    # Steady noise under the threshold raises the floor instead of counting as speech,
    # and once it has, the threshold sits well above the noise.
    assert all(v.feed(450) is None for _ in range(60))
    assert v.threshold > 450 * 2
    # Speech on top of it still gets through.
    v.feed(4000)
    assert v.feed(4000) == "started"
    # A line that is loud from the first second reads as a voice at first, then settles:
    # the floor climbs while it is on, and the level stops clearing the threshold.
    w = listen.Vad()
    edges = [w.feed(900) for _ in range(400)]
    assert edges[1] == "started" and "stopped" in edges[2:] and not w.on


def _pcm(seconds: float, *, user: float, assistant: float, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(1)
    left = user * np.sin(2 * np.pi * 220 * t) if user else rng.normal(0, 40, n)
    right = assistant * np.sin(2 * np.pi * 330 * t) if assistant else rng.normal(0, 40, n)
    return np.stack([left, right], axis=1).astype("<i2").tobytes()


def test_the_stream_reports_each_side_with_the_time_it_happened():
    edges: list[tuple[str, str, float]] = []
    clock = {"t": 1_000.0}

    def tick() -> float:
        clock["t"] += listen.WINDOW_S  # one window of audio per clock call, as in real time
        return clock["t"]

    s = listen.Stream(lambda role, status, at: edges.append((role, status, at)), clock=tick)
    assert s.header('{"type":"start","encoding":"linear16","container":"raw","sampleRate":16000,"channels":2}')["channels"] == 2
    audio = _pcm(1.0, user=0, assistant=0) + _pcm(1.5, user=0, assistant=5000) + _pcm(1.0, user=0, assistant=0) \
        + _pcm(1.2, user=4000, assistant=0) + _pcm(1.0, user=0, assistant=0)
    # Frames arrive in odd sizes; the stream reassembles windows.
    for i in range(0, len(audio), 1234):
        s.feed(audio[i:i + 1234])
    assert [(r, st) for r, st, _ in edges] == [("assistant", "started"), ("assistant", "stopped"), ("user", "started"), ("user", "stopped")]
    at = {(r, st): t for r, st, t in edges}
    # The agent spoke from 1.0 s to 2.5 s of audio, the caller from 3.5 s to 4.7 s; the
    # detector's delay is taken off, so the reported times land within a window of that.
    assert abs(at[("assistant", "started")] - 1001.0) <= 0.11 and abs(at[("assistant", "stopped")] - 1002.5) <= 0.11
    assert abs(at[("user", "started")] - 1003.5) <= 0.11 and abs(at[("user", "stopped")] - 1004.7) <= 0.11


def test_a_mono_stream_cannot_name_the_speaker():
    edges: list[tuple[str, str, float]] = []
    s = listen.Stream(lambda *e: edges.append(e))
    s.header('{"sampleRate": 16000, "channels": 1}')
    assert s.channels == 1 and s.window_bytes == 3200
    loud = (5000 * np.sin(np.arange(16000) / 7)).astype("<i2").tobytes()
    s.feed(loud)
    assert s.windows == 10 and edges == []
    assert s.header("not json") == {}  # a header we do not understand changes nothing
