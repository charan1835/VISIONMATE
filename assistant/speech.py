"""
SpeechAnnouncer — VisionMate TTS module.

Design decisions:
- pyttsx3 on Windows (SAPI5) freezes when engine.runAndWait() is called
  more than once on the same engine instance. The reliable fix is to
  create a fresh engine instance for EVERY utterance.
- Priority system: priority 0 = assistant command responses (always heard
  immediately); priority 1 = ambient object detection chatter (dropped
  whenever a command response is pending).
- A single background daemon thread drains the pending list in priority
  order. The main thread never touches pyttsx3 directly.
"""

import threading
import sys
import time
import itertools


class SpeechAnnouncer:
    """Speaks short announcements from a background worker thread."""

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        repeat_interval: float = 5.0,
        inter_utterance_delay: float = 1.5,
    ):
        self._rate = rate
        self._volume = max(0.0, min(1.0, volume))
        self._pending = []                        # List of (priority, seq, text)
        self._lock = threading.Lock()
        self._has_item = threading.Event()
        self._sequence = itertools.count()
        self._running = True
        self._inter_utterance_delay = max(0.0, float(inter_utterance_delay))

        # Object-announcement throttle state
        self._last_seen: dict = {}
        self._last_announced: dict = {}
        self._repeat_interval = max(0.0, float(repeat_interval))
        self._absence_timeout = 2.0

        # --- STOP / MUTE state ---
        # When muted, all ambient narration is suppressed.
        # Only the next explicit user command response (priority 0) will unmute.
        self._muted = False
        self._current_engine = None

        self._engine_ready = threading.Event()
        self._worker = threading.Thread(
            target=self._speech_worker,
            daemon=True,
        )
        self._worker.start()
        # Wait for the startup greeting to confirm TTS is functional
        self._engine_ready.wait(timeout=8)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, priority: int, text: str) -> None:
        with self._lock:
            self._pending.append((priority, next(self._sequence), text))
        self._has_item.set()

    def _dequeue_next(self):
        """
        Return and remove the highest-priority (lowest number) pending item.
        Blocks until one is available or _running becomes False.
        Returns None when shutting down.
        """
        while self._running:
            self._has_item.wait(timeout=0.3)
            with self._lock:
                if self._pending:
                    # Lowest priority number wins; tie-break by insertion order (seq)
                    self._pending.sort(key=lambda x: (x[0], x[1]))
                    item = self._pending.pop(0)
                    if not self._pending:
                        self._has_item.clear()
                    return item[2]   # text only
                else:
                    self._has_item.clear()
        return None

    def _speak_once(self, text: str) -> None:
        """
        Create a fresh pyttsx3 engine, speak text, then destroy it.
        Re-initialising per utterance is the only reliable pattern for
        Windows SAPI5 — reusing the same instance across multiple
        runAndWait() calls causes the second call to hang silently.
        """
        with self._lock:
            if self._muted:
                return

        try:
            import pyttsx3
            engine = pyttsx3.init()
            with self._lock:
                if self._muted:
                    try:
                        engine.stop()
                    except Exception:
                        pass
                    return
                self._current_engine = engine

            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)

            def on_word(name, location, length):
                if self._muted or not self._running:
                    try:
                        engine.stop()
                    except Exception:
                        pass

            try:
                engine.connect("started-word", on_word)
            except Exception:
                pass

            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
        except Exception as exc:
            print(f"[Speech] Error speaking text: {exc}")
        finally:
            with self._lock:
                self._current_engine = None

    def _speech_worker(self) -> None:
        # On Windows, COM must be initialised on the thread that uses SAPI5.
        pythoncom = None
        com_initialized = False
        if sys.platform == "win32":
            try:
                import pythoncom as _pythoncom
                _pythoncom.CoInitialize()
                pythoncom = _pythoncom
                com_initialized = True
            except Exception as exc:
                print(f"[Speech] COM init failed: {exc}")

        # Startup greeting — also validates that TTS works at all.
        try:
            self._speak_once("VisionMate speech is ready")
            print("[Speech] Text-to-speech engine started successfully.")
        except Exception as exc:
            print(f"[Speech] Startup greeting failed: {exc}")
        finally:
            self._engine_ready.set()

        # Main loop — one fresh engine instance per utterance.
        try:
            while self._running:
                text = self._dequeue_next()
                if text:
                    self._speak_once(text)
                    if self._inter_utterance_delay > 0:
                        time.sleep(self._inter_utterance_delay)
        finally:
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mute(self) -> None:
        """
        Silence all ambient narration immediately and abort current speech playback.
        Called when user says 'stop' or presses 'S'.
        Unmuted automatically on the next priority-0 (assistant command) announcement.
        """
        with self._lock:
            # Flush ALL queued items — both ambient and any pending P0 items
            self._pending = []
            self._has_item.clear()
            self._muted = True
            active_engine = self._current_engine

        # Immediately stop currently active audio if an engine is mid-utterance
        if active_engine is not None:
            try:
                active_engine.stop()
            except Exception:
                pass

        print("[Speech] Muted — narration halted immediately.")

    def announce(self, text: str) -> None:
        """
        Speak an assistant command response (priority 0).
        Clears any queued ambient announcements so the user hears the
        response without delay.
        Calling this also unmutes the announcer if it was previously stopped.
        """
        if not text:
            return
        announcement = str(text).strip()
        with self._lock:
            # Drop all background (priority 1) items; keep any pending command responses
            self._pending = [item for item in self._pending if item[0] == 0]
            # Unmute — user issued a new command so narration may resume after this
            self._muted = False
        self._enqueue(0, announcement)

    def _announce_background(self, text: str) -> None:
        """Queue a low-priority ambient object detection announcement (suppressed when muted)."""
        if text and not self._muted:
            self._enqueue(1, str(text).strip())

    def announce_objects(self, labels) -> None:
        """
        Announce each newly visible object class.
        Throttled by repeat_interval so the same label is not spammed.
        """
        now = time.monotonic()
        current_labels = {
            str(label).strip().lower()
            for label in labels
            if str(label).strip()
        }

        new_labels = []
        for label in current_labels:
            last_announced = self._last_announced.get(label)
            if (
                label not in self._last_seen
                or last_announced is None
                or now - last_announced >= self._repeat_interval
            ):
                new_labels.append(label)
            self._last_seen[label] = now

        # Prune labels that have disappeared from view
        self._last_seen = {
            label: ts
            for label, ts in self._last_seen.items()
            if now - ts <= self._absence_timeout
        }
        self._last_announced = {
            label: ts
            for label, ts in self._last_announced.items()
            if label in self._last_seen
        }

        if new_labels:
            new_labels.sort()
            self._announce_background(" and ".join(new_labels) + " detected")
            for label in new_labels:
                self._last_announced[label] = now

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._running = False
        self._has_item.set()          # Wake the worker so it can exit cleanly
        self._worker.join(timeout=2.0)
