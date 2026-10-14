"""Who is speaking, from the call's live audio.

On GPT-Live, Vapi sends the server no ``speech-update`` and no ``transcript`` while the call
is on (only tool calls and the end-of-call report), so on a phone call nothing tells the
page that the caller or the agent has started talking. Vapi does stream every call's audio
live: the call's ``monitor.listenUrl`` is a WebSocket that sends a JSON header and then raw
PCM, 16-bit little-endian, 16 kHz, two channels (the caller on channel 0, the agent on
channel 1, as Vapi's transcriber streams are laid out). This module connects to it, measures
the loudness of each channel in 100 ms windows, and reports a speech start after 200 ms of
sound and a stop after 700 ms of quiet, per side, with the time it happened. That is what the
page needs to light the boxes, and what the timeline's lanes are made of.

The stream carries no words. Those still come from the browser SDK (web calls) and from
``artifact.messages`` at each tool call and at hang-up (every call), see ``Agent.note_turns``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger("voice.listen")

WINDOW_S = 0.1  # loudness is measured per 100 ms
START_WINDOWS = 2  # 200 ms of sound starts a turn
STOP_WINDOWS = 7  # 700 ms of quiet ends it (a breath between sentences is shorter)
FLOOR_MIN = 200.0  # int16 RMS below which it is the line's own noise
FLOOR_CAP = 2 * FLOOR_MIN  # the floor never rises past this, so soft speech still clears the threshold
ROLES = {0: "user", 1: "assistant"}

SpeechCallback = Callable[[str, str, float], None]  # (role, "started" | "stopped", at)


class Vad:
    """Energy-based voice activity for one channel: an adaptive noise floor (follows the
    quiet level down fast and up slowly, so a noisy phone line does not read as speech)
    and hysteresis on both edges. ``feed`` takes one window's RMS and returns the edge it
    crossed, if any."""

    def __init__(self) -> None:
        self.floor = FLOOR_MIN
        self.on = False
        self.loud = 0
        self.quiet = 0

    @property
    def threshold(self) -> float:
        return max(2.5 * FLOOR_MIN, 3.0 * self.floor)

    def feed(self, rms: float) -> str | None:
        if rms < self.floor:
            self.floor = max(FLOOR_MIN, 0.8 * self.floor + 0.2 * rms)  # down fast
        elif self.on:
            self.floor = min(0.99 * self.floor + 0.01 * rms, FLOOR_CAP)  # a line loud for long is a loud line
        elif rms < self.threshold:
            self.floor = min(0.98 * self.floor + 0.02 * rms, FLOOR_CAP)  # up slowly
        if rms >= self.threshold:
            self.loud += 1
            self.quiet = 0
        else:
            self.quiet += 1
            self.loud = 0
        if not self.on and self.loud >= START_WINDOWS:
            self.on = True
            return "started"
        if self.on and self.quiet >= STOP_WINDOWS:
            self.on = False
            return "stopped"
        return None


class Stream:
    """Turns the bytes of the listen stream into speech edges. Feed it the JSON header
    first (``header``), then every binary frame (``feed``); it calls back with
    ``(role, status, at)`` where ``at`` is the epoch second the speech actually began or
    ended (the detector's delay is subtracted)."""

    def __init__(self, on_speech: SpeechCallback, *, clock: Callable[[], float] = time.time) -> None:
        self.on_speech = on_speech
        self.clock = clock
        self.sample_rate = 16000
        self.channels = 2
        self.buf = bytearray()
        self.vad = {ch: Vad() for ch in range(2)}
        self.windows = 0

    def header(self, text: str) -> dict[str, Any]:
        try:
            info = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(info, dict):
            return {}
        self.sample_rate = int(info.get("sampleRate") or info.get("sample_rate") or self.sample_rate)
        self.channels = int(info.get("channels") or self.channels)
        self.vad = {ch: Vad() for ch in range(self.channels)}
        return info

    @property
    def window_bytes(self) -> int:
        return int(self.sample_rate * WINDOW_S) * self.channels * 2

    def feed(self, data: bytes) -> list[tuple[str, str, float]]:
        self.buf += data
        edges: list[tuple[str, str, float]] = []
        n = self.window_bytes
        while len(self.buf) >= n:
            window = bytes(self.buf[:n])
            del self.buf[:n]
            self.windows += 1
            pcm = np.frombuffer(window, dtype="<i2").astype(np.float32)
            frames = pcm.reshape(-1, self.channels)
            now = self.clock()
            for ch in range(self.channels):
                rms = float(np.sqrt(np.mean(frames[:, ch] ** 2))) if len(frames) else 0.0
                edge = self.vad[ch].feed(rms)
                if edge is None:
                    continue
                role = ROLES.get(ch) if self.channels > 1 else None
                if role is None:
                    continue  # a mono stream cannot tell the two sides apart
                at = now - (START_WINDOWS if edge == "started" else STOP_WINDOWS) * WINDOW_S
                edges.append((role, edge, at))
                self.on_speech(role, edge, at)
        return edges


async def listen(call_id: str, url: str, on_speech: SpeechCallback, *, on_open: Callable[[], None] | None = None) -> None:
    """Connect to the call's listen stream and report speech edges until the call ends
    (Vapi closes the socket). Errors end the listener quietly: the page then has what the
    other paths give it, as before."""
    import websockets

    stream = Stream(on_speech)
    try:
        async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
            if on_open:
                on_open()
            async for frame in ws:
                if isinstance(frame, bytes):
                    stream.feed(frame)
                else:
                    info = stream.header(str(frame))
                    log.info("listen %s: %s", call_id, {k: info.get(k) for k in ("type", "encoding", "sampleRate", "channels")} if info else frame[:120])
                    if info and stream.channels < 2:
                        log.warning("listen %s: a mono stream; the two sides cannot be told apart", call_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the listener is a nicety
        log.info("listen %s ended: %s: %s", call_id, type(exc).__name__, exc)
    finally:
        log.info("listen %s: %d windows", call_id, stream.windows)


__all__ = ["FLOOR_CAP", "FLOOR_MIN", "START_WINDOWS", "STOP_WINDOWS", "WINDOW_S", "Stream", "Vad", "listen"]
