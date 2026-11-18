import queue
import threading
import sys
import time


class SpeechAnnouncer:
    """Speaks short announcements from a background worker."""

    def __init__(self, rate=175, volume=1.0):
        self._queue = queue.Queue()
        self._running = True
        self._seen_labels = {}
        self._absence_timeout = 2.0
        self._engine_ready = threading.Event()
        self._worker = threading.Thread(
            target=self._speech_worker,
            args=(rate, volume),
            daemon=True,
        )
        self._worker.start()
        self._engine_ready.wait(timeout=3)

    def _speech_worker(self, rate, volume):
        pythoncom = None
        com_initialized = False
        try:
            import pyttsx3

            if sys.platform == "win32":
                import pythoncom
                pythoncom.CoInitialize()
                com_initialized = True

            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", max(0.0, min(1.0, volume)))

            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
                print(f"[Speech] Using voice: {voices[0].name}")

            engine.say("VisionMate speech is ready")
            engine.runAndWait()
            self._engine_ready.set()
            print("[Speech] Text-to-speech engine started")
        except Exception as exc:
            print(f"Speech unavailable: {exc}")
            self._engine_ready.set()
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()
            return

        try:
            while self._running:
                try:
                    text = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if text:
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as exc:
                        print(f"Speech error: {exc}")
                self._queue.task_done()
        finally:
            if com_initialized and pythoncom is not None:
                pythoncom.CoUninitialize()

    def announce(self, text):
        if text:
            announcement = str(text).strip()
            self._queue.put(announcement)

    def announce_objects(self, labels):
        """Announce each object type when it first becomes visible."""
        now = time.monotonic()
        current_labels = {
            str(label).strip().lower()
            for label in labels
            if str(label).strip()
        }

        new_labels = [
            label for label in sorted(current_labels)
            if label not in self._seen_labels
        ]
        for label in current_labels:
            self._seen_labels[label] = now

        self._seen_labels = {
            label: last_seen
            for label, last_seen in self._seen_labels.items()
            if now - last_seen <= self._absence_timeout
        }

        if new_labels:
            self.announce(" and ".join(new_labels) + " detected")

    def stop(self):
        self._running = False
        self._worker.join(timeout=1.0)
