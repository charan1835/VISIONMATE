"""
VisionMate - Main Application Entry Point

Connects real-time webcam perception (YOLO26n + ByteTrack + Track Recovery + 2D Position + Proximity)
with non-blocking terminal natural-language command processing (ObjectCommandParser +
ObjectResponseGenerator + AssistantController) and desktop GUI visual feedback.
"""

import sys
import os
import time
import queue
import threading
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from detection.tracker import ObjectTracker
from detection.recovery import TrackRecoveryManager
from detection.stability import TemporalDetectionStabilizer
from spatial.position import ObjectPositionAnalyzer
from spatial.distance import DistanceEstimator
from spatial.obstacle import ObstacleAnalyzer
from assistant.controller import AssistantController
from assistant.query_memory import ObjectQueryMemory
from assistant.speech import SpeechAnnouncer


def find_working_camera():
    """
    Test camera indices 0, 1, 2, 3 in sequence and return the first working VideoCapture.
    """
    for index in [0, 1, 2, 3]:
        print(f"Testing camera index: {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"--> Using camera index: {index}")
                return cap, index
            cap.release()
    return None, -1


def draw_track_label(frame, track):
    """Draw a high-contrast, three-line label for a tracked object."""
    x1, y1, x2, y2 = track["bbox"]
    track_id = track["track_id"]
    class_name = str(track["class_name"]).replace("_", " ")
    confidence = track["confidence"]
    is_stale = track.get("is_stale", False)
    is_recovered = track.get("recovered", False)

    box_color = (0, 165, 255) if is_stale else ((255, 255, 0) if is_recovered else (0, 255, 0))
    status_tag = " STALE" if is_stale else (" RECOVERED" if is_recovered else "")

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    horiz = str(track["position"]["horizontal"]).upper()
    vert = str(track["position"]["vertical"]).upper()
    dist_cat = str(track["distance"]["category"]).upper().replace("_", " ")
    lines = [
        f"ID {track_id} | {class_name}{status_tag} | {confidence:.2f}",
        f"POS: {horiz} / {vert}",
        f"DIST: {dist_cat}",
    ]

    frame_h, frame_w = frame.shape[:2]
    font_scale = 0.52 if frame_w >= 1280 else 0.44
    thickness = 2
    line_height = 24 if frame_w >= 1280 else 21
    max_width = 0
    text_sizes = []
    for line in lines:
        size, baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_sizes.append((size, baseline))
        max_width = max(max_width, size[0])

    label_width = max_width + 16
    label_height = len(lines) * line_height + 8
    label_x1 = max(0, min(x1, frame_w - label_width))
    label_y2 = max(y1 - 6, label_height + 6)
    label_y1 = label_y2 - label_height

    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (min(frame_w - 1, label_x1 + label_width), label_y2),
        (15, 15, 15),
        -1,
    )
    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (min(frame_w - 1, label_x1 + label_width), label_y2),
        box_color,
        1,
    )

    for index, line in enumerate(lines):
        text_y = label_y1 + 18 + index * line_height
        text_color = box_color if index == 0 else ((255, 255, 0) if index == 1 else (0, 255, 255))
        cv2.putText(
            frame,
            line,
            (label_x1 + 8, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (label_x1 + 8, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )


def input_listener(command_queue: queue.Queue, is_running: list):
    """
    Secondary thread listening for microphone voice commands without freezing the camera loop.
    """
    print("\n----------------------------------------------------------")
    print("  VisionMate Voice Assistant Active")
    print("  Speak a command (e.g. 'Find my bottle').")
    print("  Press 'Q' or 'ESC' in the camera window to exit.")
    print("  Press 'F' in the camera window to toggle fullscreen.")
    print("----------------------------------------------------------\n")

    try:
        import speech_recognition as sr
        import sounddevice as sd
    except Exception as exc:
        print(f"Voice command unavailable: {exc}")
        print("Text input fallback: type a command and press Enter.")
        while is_running[0]:
            try:
                user_input = input("> ")
                if user_input and user_input.strip():
                    command_queue.put(user_input.strip())
            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                pass
        return

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 0.8

    while is_running[0]:
        try:
            print("[Voice] Listening...")
            sample_rate = 48000
            audio_bytes = sd.rec(
                int(sample_rate * 4),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            audio = sr.AudioData(
                audio_bytes.tobytes(),
                sample_rate,
                2,
            )
            command = recognizer.recognize_google(audio, language="en-US").strip()
            if command:
                print(f"[Voice Command Received] '{command}'")
                command_queue.put(command)
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            print(f"Voice recognition service unavailable: {exc}")
        except Exception:
            pass


def render_obstacle_overlay(frame, warning: dict):
    if not warning or not isinstance(warning, dict):
        return

    frame_h, frame_w = frame.shape[:2]
    obj_str = str(warning.get("object", "")).upper()
    pos_str = str(warning.get("position", "")).upper()
    dist_str = str(warning.get("distance", "")).upper().replace("_", " ")
    severity = str(warning.get("severity", "warning")).lower()

    text = f"{obj_str} - {pos_str} - {dist_str}"

    if severity == "critical":
        bg_color = (0, 0, 160)
        border_color = (0, 0, 255)
        text_color = (0, 255, 255)
        header_text = "[CRITICAL OBSTACLE]"
    else:
        bg_color = (0, 75, 150)
        border_color = (0, 165, 255)
        text_color = (255, 255, 255)
        header_text = "[OBSTACLE WARNING]"

    box_w = max(320, int(len(text) * 11) + 20)
    x2 = frame_w - 10
    x1 = max(230, x2 - box_w)
    y1 = 10
    y2 = 65

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    cv2.putText(frame, header_text, (x1 + 10, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, border_color, 2)
    cv2.putText(frame, text, (x1 + 10, y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1)


def render_response_overlay(frame: np.ndarray, response_text: str):
    """
    Render a multi-line formatted assistant response box at the bottom of the frame.
    """
    if not response_text:
        return

    frame_h, frame_w = frame.shape[:2]
    max_text_w = frame_w - 40

    # Wrap words into multiline list
    words = response_text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if w > max_text_w and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    # Calculate shaded box height
    line_h = 24
    header_h = 25
    box_h = len(lines) * line_h + header_h + 10

    y1 = frame_h - box_h - 10
    y2 = frame_h - 10

    # Draw dark background box with cyan border
    cv2.rectangle(frame, (10, y1), (frame_w - 10, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, y1), (frame_w - 10, y2), (255, 255, 0), 2)

    # Title label
    cv2.putText(frame, "VisionMate Assistant:", (20, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # Render wrapped response lines
    for idx, line in enumerate(lines):
        cv2.putText(frame, line, (20, y1 + 20 + header_h + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def main():
    print("==========================================================")
    print("               VisionMate Application v1.0                ")
    print("==========================================================")

    window_name = "VisionMate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True

    # 1. Probe & Open Physical Camera
    cap, cam_idx = find_working_camera()
    if cap is None:
        print("\nERROR: Could not open any webcam (tried indices 0, 1, 2, 3).")
        print("Please verify camera connection and device permissions.")
        sys.exit(1)

    is_running = [True]
    command_queue = queue.Queue()

    try:
        # Determine actual frame resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width <= 0 or frame_height <= 0:
            ret, sample_frame = cap.read()
            if ret and sample_frame is not None:
                frame_height, frame_width = sample_frame.shape[:2]
            else:
                frame_width, frame_height = 640, 480

        print(f"Webcam active at resolution: {frame_width}x{frame_height}")

        # 2. Initialize Perception & Assistant Modules
        print("Initializing YOLO11n baseline + ByteTrack...")
        project_root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_root, "yolo11n.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, "yolo26n.pt")

        tracker = ObjectTracker(
            model_path_or_detector=model_path,
            confidence=0.35,
            tracker_type=os.path.join(project_root, "models", "visionmate_bytetrack.yaml"),
            imgsz=640,
        )
        recovery_manager = TrackRecoveryManager(
            max_recovery_frames=15,
            max_center_distance=0.20,
            min_size_ratio=0.5,
            max_size_ratio=2.0,
            min_recovery_score=0.70,
            frame_width=frame_width,
            frame_height=frame_height
        )
        stabilizer = TemporalDetectionStabilizer(max_missing_frames=8, class_change_threshold=3)

        position_analyzer = ObjectPositionAnalyzer(frame_width=frame_width, frame_height=frame_height)
        distance_estimator = DistanceEstimator(frame_width=frame_width, frame_height=frame_height)
        obstacle_analyzer = ObstacleAnalyzer(
            min_confidence=0.40,
            warning_distance="close",
            critical_distance="very_close"
        )
        query_memory = ObjectQueryMemory(max_history_frames=30)
        assistant_controller = AssistantController(query_memory=query_memory)

        print("All system components loaded successfully.")

        # 3. Start Non-Blocking Input Listener Thread
        input_thread = threading.Thread(
            target=input_listener,
            args=(command_queue, is_running),
            daemon=True
        )
        input_thread.start()

        fps = 0.0
        prev_time = time.time()
        active_response = ""
        speech = SpeechAnnouncer(rate=175, volume=1.0)
        minimum_speech_confidence = 0.35

        # 4. Main Perception & Control Loop
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Failed to read frame from webcam.")
                break

            # Live FPS calculation
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 0.001))
            prev_time = curr_time

            # A. Detection & ByteTrack Tracking
            raw_tracks = tracker.track(frame)

            # B. Apply Track Recovery Layer (recovers short-gap ID fragmentation)
            recovered_tracks = recovery_manager.update(raw_tracks)

            # C. Apply Temporal Stability Layer (retains brief misses up to 8 frames)
            stable_tracks = stabilizer.update(recovered_tracks)

            # D. 2D Position & Relative Distance Spatial Analysis
            current_detections = []
            for track in stable_tracks:
                # Calculate fresh position/distance if not already attached
                if "position" not in track:
                    track = position_analyzer.analyze_track(track)
                if "distance" not in track:
                    track = distance_estimator.analyze_track(track)
                current_detections.append(track)

            # Update query memory on every frame
            query_memory.update(current_detections)

            # Analyze proximity obstacles
            obstacle_warnings = obstacle_analyzer.analyze(current_detections)

            # Announce newly visible object types and repeat them periodically.
            speech_labels = [
                track["class_name"]
                for track in current_detections
                if not track.get("is_stale", False)
                and float(track.get("confidence", 0.0)) >= minimum_speech_confidence
            ]
            speech.announce_objects(speech_labels)

            # E. Check for New User Commands (Non-blocking queue check)
            try:
                cmd_text = command_queue.get_nowait()
                if cmd_text:
                    print(f"\n[Command Received] '{cmd_text}'")
                    active_response = assistant_controller.process_command(cmd_text, current_detections)
                    print(f"[Assistant Feedback] '{active_response}'\n")
                    speech.announce(active_response)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing command: {e}")

            # F. Render Visual Bounding Boxes & Labels
            for t in current_detections:
                draw_track_label(frame, t)

            # G. Top-Left Header Box
            cv2.rectangle(frame, (10, 10), (220, 75), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (220, 75), (0, 255, 0), 1)
            cv2.putText(frame, "VisionMate App", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Objects: {len(current_detections)}", (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # H. Top-Right Obstacle Warning & Bottom Assistant Response Overlays
            if obstacle_warnings:
                render_obstacle_overlay(frame, obstacle_warnings[0])

            if active_response:
                render_response_overlay(frame, active_response)

            # I. Display Window
            cv2.imshow(window_name, frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 27 = ESC
                print("\nExit key pressed. Stopping VisionMate...")
                break
            if key == ord('f') or key == ord('F'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
    except Exception as e:
        print(f"\nUnexpected error in app loop: {e}")
    finally:
        is_running[0] = False
        if "speech" in locals():
            speech.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("VisionMate shut down cleanly.")


if __name__ == "__main__":
    main()
