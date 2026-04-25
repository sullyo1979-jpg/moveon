import os
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = str(exc)
else:
    _CV2_IMPORT_ERROR = ""

# Lazy import de ultralytics (torch pese ~11s d import). On ne le charge que
# si le backend YOLO est reellement utilise.
solutions = None
_YOLO_IMPORT_ERROR = ""
_YOLO_IMPORT_ATTEMPTED = False


def _ensure_ultralytics() -> bool:
    """Charge ultralytics a la demande. Retourne True si dispo."""
    global solutions, _YOLO_IMPORT_ERROR, _YOLO_IMPORT_ATTEMPTED
    if solutions is not None:
        return True
    if _YOLO_IMPORT_ATTEMPTED:
        return False
    _YOLO_IMPORT_ATTEMPTED = True
    try:
        from ultralytics import solutions as _sol
        solutions = _sol
        return True
    except Exception as exc:
        _YOLO_IMPORT_ERROR = str(exc)
        return False

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmark,
        PoseLandmarker,
        PoseLandmarkerOptions,
        PoseLandmarksConnections,
    )
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
        VisionTaskRunningMode,
    )
except Exception as exc:
    mp = None
    BaseOptions = None
    PoseLandmarker = None
    PoseLandmarkerOptions = None
    PoseLandmark = None
    PoseLandmarksConnections = None
    VisionTaskRunningMode = None
    _MEDIAPIPE_IMPORT_ERROR = str(exc)
else:
    _MEDIAPIPE_IMPORT_ERROR = ""

try:
    import pygame
except Exception as exc:
    pygame = None
    _PYGAME_IMPORT_ERROR = str(exc)
else:
    _PYGAME_IMPORT_ERROR = ""

try:
    import pyttsx3
except Exception as exc:
    pyttsx3 = None
    _TTS_IMPORT_ERROR = str(exc)
else:
    _TTS_IMPORT_ERROR = ""


def detect_available_cameras(max_index: int = 5) -> List[Tuple[int, str]]:
    """Detecte les cameras disponibles sur le systeme.
    Retourne une liste de (index, nom_convivial).

    Detection rapide (<200ms) — ne tente PAS d ouvrir les cameras
    (car cv2.VideoCapture sur un index inexistant peut bloquer 2-3s).
    On fait confiance aux API systeme :
      1. pygrabber (DirectShow) = liste autoritaire des indices cv2
      2. WMI PowerShell = complement si pygrabber manque des cameras
    """
    # 1) Noms via pygrabber — ordre = indices cv2
    dshow_names: List[str] = []
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        dshow_names = list(FilterGraph().get_input_devices() or [])
    except Exception:
        dshow_names = []

    # 2) Noms via WMI (PowerShell) — detecte largement sous Windows
    wmi_names: List[str] = []
    if os.name == "nt":
        try:
            import subprocess
            ps_cmd = (
                "Get-CimInstance Win32_PnPEntity | "
                "Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' } | "
                "Select-Object -ExpandProperty Name"
            )
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=3,
            )
            if out.returncode == 0:
                wmi_names = [
                    line.strip() for line in out.stdout.splitlines() if line.strip()
                ]
        except Exception:
            wmi_names = []

    # 3) Merge : pygrabber d abord (index fiable), WMI complete
    results: List[Tuple[int, str]] = []
    seen_names = set()
    for i, name in enumerate(dshow_names):
        results.append((i, name))
        seen_names.add(name.lower())

    # WMI : ajoute les cameras non listees par pygrabber (indices fictifs apres)
    next_idx = len(dshow_names)
    for name in wmi_names:
        if name.lower() not in seen_names:
            results.append((next_idx, f"{name} (a tester)"))
            seen_names.add(name.lower())
            next_idx += 1

    # 4) Fallback : si tout a echoue, proposer au moins l index 0
    if not results:
        results.append((0, "Camera par defaut"))

    return results


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_HEEL = 29
MP_RIGHT_HEEL = 30
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32

MEDIAPIPE_DEFAULT_MODEL = "pose_landmarker_lite.task"
MEDIAPIPE_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MEDIAPIPE_FULL_MODEL = "pose_landmarker_full.task"
MEDIAPIPE_FULL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

YOLO_COCO_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

SQUAT_FOOT_SHOULDER_RATIO_THRESHOLDS = (1.2, 2.8)
SQUAT_KNEE_FOOT_RATIO_THRESHOLDS = {
    "up": (0.5, 1.0),
    "middle": (0.7, 1.0),
    "down": (0.7, 1.1),
}
LUNGE_KNEE_ANGLE_THRESHOLD = (60.0, 125.0)  # Inspired by repo lunge.py
MULTI_PLAYER_COLORS = [
    (72, 208, 255),
    (118, 235, 126),
    (255, 190, 92),
    (222, 126, 255),
]


def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _compute_distance_2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _mp_point(
    landmarks: Optional[list[Any]],
    idx: int,
    *,
    min_visibility: float = 0.5,
) -> Optional[np.ndarray]:
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    if lm.x is None or lm.y is None:
        return None
    visibility = 1.0 if getattr(lm, "visibility", None) is None else float(lm.visibility)
    presence = 1.0 if getattr(lm, "presence", None) is None else float(lm.presence)
    if visibility < min_visibility or presence < min_visibility:
        return None
    return np.array([float(lm.x), float(lm.y)], dtype=np.float32)


def _mp_angle(
    landmarks: Optional[list[Any]],
    a_idx: int,
    b_idx: int,
    c_idx: int,
    *,
    min_visibility: float = 0.5,
) -> Optional[float]:
    a = _mp_point(landmarks, a_idx, min_visibility=min_visibility)
    b = _mp_point(landmarks, b_idx, min_visibility=min_visibility)
    c = _mp_point(landmarks, c_idx, min_visibility=min_visibility)
    if a is None or b is None or c is None:
        return None
    return _compute_angle(a, b, c)


def _avg_valid(*values: Optional[float]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _stage_from_angle(angle: Optional[float], *, down: float, up: float) -> str:
    if angle is None:
        return "UNKNOWN"
    if angle <= down:
        return "DOWN"
    if angle >= up:
        return "UP"
    return "MID"


def _progress_from_angle(angle: Optional[float], *, down: float, up: float) -> float:
    return _compute_progress_percent(angle, down, up)


def _squat_placement_feedback(
    landmarks: Optional[list[Any]],
    *,
    stage_key: str,
    min_visibility: float,
) -> tuple[str, str]:
    # Adapted from the source project's squat.py ratio checks (feet/shoulder and knee/foot).
    ls = _mp_point(landmarks, MP_LEFT_SHOULDER, min_visibility=min_visibility)
    rs = _mp_point(landmarks, MP_RIGHT_SHOULDER, min_visibility=min_visibility)
    lk = _mp_point(landmarks, MP_LEFT_KNEE, min_visibility=min_visibility)
    rk = _mp_point(landmarks, MP_RIGHT_KNEE, min_visibility=min_visibility)
    lf = _mp_point(landmarks, MP_LEFT_FOOT_INDEX, min_visibility=min_visibility)
    rf = _mp_point(landmarks, MP_RIGHT_FOOT_INDEX, min_visibility=min_visibility)
    if any(p is None for p in (ls, rs, lk, rk, lf, rf)):
        return ("Pieds: n/a", "Genoux: n/a")

    shoulder_w = max(_compute_distance_2d(ls, rs), 1e-6)
    foot_w = max(_compute_distance_2d(lf, rf), 1e-6)
    knee_w = max(_compute_distance_2d(lk, rk), 1e-6)

    foot_ratio = foot_w / shoulder_w
    knee_foot_ratio = knee_w / foot_w

    foot_min, foot_max = SQUAT_FOOT_SHOULDER_RATIO_THRESHOLDS
    if foot_ratio < foot_min:
        foot_msg = "Pieds trop serres"
    elif foot_ratio > foot_max:
        foot_msg = "Pieds trop ecartes"
    else:
        foot_msg = "Pieds OK"

    knee_min, knee_max = SQUAT_KNEE_FOOT_RATIO_THRESHOLDS.get(
        stage_key, SQUAT_KNEE_FOOT_RATIO_THRESHOLDS["middle"]
    )
    if knee_foot_ratio < knee_min:
        knee_msg = "Genoux trop serres"
    elif knee_foot_ratio > knee_max:
        knee_msg = "Genoux trop ouverts"
    else:
        knee_msg = "Genoux OK"
    return (foot_msg, knee_msg)


def _plank_posture_metrics(
    landmarks: Optional[list[Any]],
    *,
    min_visibility: float,
) -> tuple[str, Optional[float], Optional[float], Optional[str]]:
    # Inspired by the source plank detector classes (correct / high back / low back),
    # but implemented with lightweight geometry instead of ML models.
    candidates = [
        ("left", MP_LEFT_SHOULDER, MP_LEFT_HIP, MP_LEFT_ANKLE),
        ("right", MP_RIGHT_SHOULDER, MP_RIGHT_HIP, MP_RIGHT_ANKLE),
    ]

    best: Optional[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = None
    for side, s_idx, h_idx, a_idx in candidates:
        s = _mp_point(landmarks, s_idx, min_visibility=min_visibility)
        h = _mp_point(landmarks, h_idx, min_visibility=min_visibility)
        a = _mp_point(landmarks, a_idx, min_visibility=min_visibility)
        if s is None or h is None or a is None:
            continue
        span = _compute_distance_2d(s, a)
        if best is None or span > best[4]:
            best = (side, s, h, a, span)

    if best is None:
        return ("UNKNOWN", None, None, None)

    side, shoulder, hip, ankle, span = best
    body_angle = _compute_angle(shoulder, hip, ankle)
    if body_angle is None:
        return ("UNKNOWN", None, None, side)

    line = ankle - shoulder
    denom = float(np.linalg.norm(line))
    if denom < 1e-6:
        return ("UNKNOWN", body_angle, None, side)
    signed_dist = float(np.cross(line, hip - shoulder) / denom)
    if float(ankle[0] - shoulder[0]) < 0:
        signed_dist *= -1.0

    if body_angle >= 160.0 and abs(signed_dist) <= 0.03:
        stage = "CORRECT"
    elif signed_dist < -0.02:
        stage = "HIGH BACK"
    elif signed_dist > 0.02:
        stage = "LOW BACK"
    else:
        stage = "ADJUST"

    return (stage, body_angle, signed_dist, side)


def _arm_state_from_angles(
    left_angle: Optional[float],
    right_angle: Optional[float],
    down_threshold: float,
    up_threshold: float,
) -> Tuple[bool, bool, Optional[float]]:
    valid_angles = [a for a in (left_angle, right_angle) if a is not None]
    avg_angle = float(np.mean(valid_angles)) if valid_angles else None

    if left_angle is not None and right_angle is not None:
        is_down = bool(left_angle <= down_threshold and right_angle <= down_threshold)
        is_up = bool(left_angle >= up_threshold and right_angle >= up_threshold)
        return is_down, is_up, avg_angle

    if avg_angle is not None:
        return bool(avg_angle <= down_threshold), bool(avg_angle >= up_threshold), avg_angle

    return False, False, None


def _compute_progress_percent(
    angle: Optional[float], down_threshold: float, up_threshold: float
) -> float:
    if angle is None:
        return 0.0
    denom = float(up_threshold - down_threshold)
    if abs(denom) < 1e-6:
        return 0.0
    percent = (up_threshold - angle) / denom * 100.0
    return float(np.clip(percent, 0.0, 100.0))


def _draw_progress_bar(
    image: np.ndarray,
    progress_percent: float,
    label: str = "Progression",
) -> None:
    h, w = image.shape[:2]
    bar_w = min(460, max(220, w - 40))
    bar_h = 18
    x = 20
    y = h - 24
    progress_percent = float(np.clip(progress_percent, 0.0, 100.0))
    fill_w = int((bar_w - 4) * (progress_percent / 100.0))

    _blend_rect(
        image,
        x,
        y - bar_h,
        x + bar_w,
        y,
        color=(12, 18, 30),
        alpha=0.62,
        border_color=(62, 92, 124),
        border_thickness=1,
    )
    if fill_w > 0:
        _blend_rect(
            image,
            x + 2,
            y - bar_h + 2,
            x + 2 + fill_w,
            y - 2,
            color=(68, 205, 255),
            alpha=0.92,
            border_color=None,
            border_thickness=0,
        )
    cv2.putText(
        image,
        f"{label}: {progress_percent:5.1f}%",
        (x + 4, y - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 235, 245),
        1,
        cv2.LINE_AA,
    )


def _draw_hud_lines(display: np.ndarray, lines: list[str]) -> None:
    for i, text in enumerate(lines):
        y = 35 + i * 35
        cv2.putText(
            display,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 230, 255),
            2,
            cv2.LINE_AA,
        )


class _VoiceCoach:
    # Phrases motivantes par jalon (reps). Le coach choisit aleatoirement pour varier.
    _MILESTONE_PHRASES = {
        5:   ["Cinq reps, super depart !", "Cinq, tu es lance !", "Cinq, on continue !"],
        10:  ["Dix reps, bravo !", "Dix, superbe rythme !", "Dix, garde le cap !"],
        15:  ["Quinze, impressionnant !", "Quinze, tu assures !"],
        20:  ["Vingt reps, tu es une machine !", "Vingt, champion !", "Vingt, force !"],
        25:  ["Vingt-cinq, top niveau !", "Vingt-cinq, on ne lache rien !"],
        30:  ["Trente reps, incroyable !", "Trente, mental d acier !"],
        40:  ["Quarante, tu depasses tes limites !"],
        50:  ["Cinquante, tu es un guerrier !", "Cinquante, moment legendaire !"],
        75:  ["Soixante-quinze, performance hors norme !"],
        100: ["Cent reps, respect total !", "Cent, tu entres dans la legende !"],
    }
    _START_PHRASES = [
        "C est parti, donne tout !",
        "On y va, reste concentre !",
        "Prepare-toi, a toi de jouer !",
        "Sois fier de bouger, c est parti !",
    ]
    _ENCOURAGE_PHRASES = [
        "Allez, continue !",
        "Tu peux le faire !",
        "Reste fort !",
        "Respire et continue !",
        "Bouge, progresse, domine !",
    ]

    def __init__(self, *, enabled: bool = True) -> None:
        self.supported = pyttsx3 is not None
        self.enabled = bool(enabled) and self.supported
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=8)
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_global = 0.0
        self._last_by_key: Dict[str, float] = {}
        self._advice_candidate_key: Optional[str] = None
        self._advice_candidate_count = 0
        self._tracking_issue_announced = False
        self._milestones_announced: set = set()
        self._last_encouragement_at = 0.0
        self._session_started = False
        if self.supported:
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    def _select_french_voice(self, engine: Any) -> None:
        try:
            voices = engine.getProperty("voices") or []
        except Exception:
            voices = []

        best_voice_id = None
        for voice in voices:
            attrs = []
            attrs.append(str(getattr(voice, "id", "")))
            attrs.append(str(getattr(voice, "name", "")))
            langs = getattr(voice, "languages", []) or []
            for lang in langs:
                if isinstance(lang, bytes):
                    try:
                        attrs.append(lang.decode("utf-8", errors="ignore"))
                    except Exception:
                        attrs.append(str(lang))
                else:
                    attrs.append(str(lang))
            hay = " ".join(attrs).lower()
            if any(tag in hay for tag in ["french", "fr-fr", "fr_", " fr ", "france", "fr"]):
                best_voice_id = getattr(voice, "id", None)
                # Prefer explicit French voices
                if "french" in hay or "fr-fr" in hay:
                    break
        if best_voice_id:
            try:
                engine.setProperty("voice", best_voice_id)
            except Exception:
                pass

    def _worker(self) -> None:
        engine = None
        try:
            engine = pyttsx3.init()
            self._select_french_voice(engine)
            try:
                engine.setProperty("rate", 145)
            except Exception:
                pass
        except Exception:
            self.supported = False
            self.enabled = False
            return

        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # Keep the app running even if TTS backend fails mid-session.
                pass

        try:
            engine.stop()
        except Exception:
            pass

    def close(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
            self._worker_thread.join(timeout=0.2)

    def status_label(self) -> str:
        if not self.supported:
            return "Voix: indisponible"
        return "Voix: ON" if self.enabled else "Voix: OFF"

    def toggle(self) -> bool:
        if not self.supported:
            self.enabled = False
            return False
        self.enabled = not self.enabled
        if self.enabled:
            self.say("Voix activée", key="voice-on", cooldown=0.1, force=True)
        return self.enabled

    def say(
        self,
        text: str,
        *,
        key: Optional[str] = None,
        cooldown: float = 2.5,
        min_global_interval: float = 0.8,
        force: bool = False,
    ) -> bool:
        if not self.supported or not self.enabled:
            return False

        clean = " ".join(str(text).strip().split())
        if not clean:
            return False
        now = time.time()
        phrase_key = str(key or clean).strip().lower()

        with self._lock:
            if not force:
                if now - self._last_global < float(min_global_interval):
                    return False
                if now - self._last_by_key.get(phrase_key, -1e9) < float(cooldown):
                    return False
            self._last_global = now
            self._last_by_key[phrase_key] = now

        try:
            self._queue.put_nowait(clean)
            return True
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except Exception:
                return False
            try:
                self._queue.put_nowait(clean)
                return True
            except Exception:
                return False

    def announce_rep(self, reps: int) -> None:
        n = int(reps)
        if n <= 0:
            return
        self.say(
            f"{n}",
            key=f"rep-{n}",
            cooldown=3600.0,
            min_global_interval=0.15,
        )
        # Jalons motivants : phrase aleatoire declenchee une seule fois par seance
        if n in self._MILESTONE_PHRASES and n not in self._milestones_announced:
            self._milestones_announced.add(n)
            import random
            phrase = random.choice(self._MILESTONE_PHRASES[n])
            self.say(
                phrase,
                key=f"milestone-{n}",
                cooldown=3600.0,
                min_global_interval=0.6,
            )

    def announce_start(self, exercise_name: str) -> None:
        """Annonce motivante au debut d'une seance."""
        if self._session_started:
            return
        self._session_started = True
        import random
        intro = random.choice(self._START_PHRASES)
        self.say(
            f"{exercise_name}. {intro}",
            key="session-start",
            cooldown=3600.0,
            min_global_interval=0.1,
            force=True,
        )

    def announce_encouragement(self, *, min_interval: float = 25.0) -> None:
        """Phrase d'encouragement periodique pendant l'effort."""
        now = time.time()
        if now - self._last_encouragement_at < min_interval:
            return
        self._last_encouragement_at = now
        import random
        phrase = random.choice(self._ENCOURAGE_PHRASES)
        self.say(
            phrase,
            key="encouragement",
            cooldown=10.0,
            min_global_interval=1.5,
        )

    def announce_completion(
        self,
        *,
        reps: int = 0,
        hold_seconds: float = 0.0,
        success: bool = True,
        challenge_active: bool = False,
    ) -> None:
        """Phrase de cloture en fin de seance."""
        if reps > 0:
            base = f"Seance terminee. {int(reps)} repetitions."
        elif hold_seconds > 0:
            base = f"Seance terminee. {hold_seconds:.0f} secondes tenues."
        else:
            base = "Seance terminee."
        if challenge_active:
            base += " Defi reussi, bravo !" if success else " Continue, tu progresses !"
        else:
            base += " Bravo, sois fier de toi !"
        self.say(
            base,
            key="session-end",
            cooldown=3600.0,
            min_global_interval=0.1,
            force=True,
        )

    def reset_session(self) -> None:
        """Reinitialise les jalons entre deux seances."""
        self._milestones_announced.clear()
        self._last_encouragement_at = 0.0
        self._session_started = False

    def update_advice(
        self,
        text: str,
        *,
        severity: str = "info",
        stable_frames: int = 12,
    ) -> None:
        sev = str(severity)
        if sev not in {"warn", "error"}:
            self._advice_candidate_key = None
            self._advice_candidate_count = 0
            self._tracking_issue_announced = False
            return

        key = " ".join(str(text).strip().split()).lower()
        if not key:
            return

        if key == self._advice_candidate_key:
            self._advice_candidate_count += 1
        else:
            self._advice_candidate_key = key
            self._advice_candidate_count = 1

        is_tracking_issue = any(
            token in key
            for token in [
                "cadre",
                "montre tout le corps",
                "place les epaules",
                "position non detectee",
                "place-toi de profil",
                "hanches, genoux et pieds",
            ]
        )

        if self._advice_candidate_count >= max(1, int(stable_frames)):
            if is_tracking_issue and self._tracking_issue_announced:
                return
            self.say(
                text,
                key=f"advice:{key}",
                cooldown=20.0 if is_tracking_issue else (4.0 if sev == "warn" else 5.0),
                min_global_interval=1.2,
            )
            if is_tracking_issue:
                self._tracking_issue_announced = True


def _make_voice_coach(config: Dict[str, Any]) -> _VoiceCoach:
    enabled = bool(config.get("voice_enabled", False))
    return _VoiceCoach(enabled=enabled)


class _MusicCoach:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.supported = pygame is not None
        self.enabled = bool(config.get("music_enabled", False))
        self.volume = float(np.clip(float(config.get("music_volume", 0.45)), 0.0, 1.0))
        self._started = False
        self._tracks = self._discover_tracks(config)
        self._track_index = 0

        if not self.supported:
            self.enabled = False
            return
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()
        except Exception:
            self.supported = False
            self.enabled = False
            return

        if self.enabled:
            self._play_current()

    def _discover_tracks(self, config: Dict[str, Any]) -> list[Path]:
        configured = config.get("music_tracks")
        candidates: list[Path] = []
        if isinstance(configured, list):
            for item in configured:
                try:
                    path = Path(str(item)).expanduser()
                except Exception:
                    continue
                if not path.is_absolute():
                    path = Path.cwd() / path
                if path.exists() and path.is_file():
                    candidates.append(path)

        if candidates:
            return candidates

        assets_dir = Path(__file__).resolve().parent / "assets" / "music"
        if not assets_dir.exists():
            return []
        return sorted(
            [p for p in assets_dir.iterdir() if p.is_file() and p.suffix.lower() in {".ogg", ".mp3", ".wav"}]
        )

    def _play_current(self) -> bool:
        if not self.supported or not self._tracks:
            return False
        try:
            track = self._tracks[self._track_index % len(self._tracks)]
            pygame.mixer.music.load(str(track))
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play(-1)
            self._started = True
            return True
        except Exception:
            self._started = False
            return False

    def toggle(self) -> bool:
        if not self.supported or not self._tracks:
            self.enabled = False
            return False
        self.enabled = not self.enabled
        if self.enabled:
            self._play_current()
        else:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self._started = False
        return self.enabled

    def close(self) -> None:
        if self._started:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._started = False

    def status_label(self) -> str:
        if not self.supported:
            return "Musique: indisponible"
        if not self._tracks:
            return "Musique: aucune piste"
        return "Musique: ON" if self.enabled else "Musique: OFF"


def _session_status_label(voice: _VoiceCoach, music: _MusicCoach) -> str:
    voice_text = "Voix ON" if voice.enabled and voice.supported else ("Voix OFF" if voice.supported else "Voix n/a")
    music_text = "Musique ON" if music.enabled and music.supported and music._tracks else (
        "Musique OFF" if music.supported and music._tracks else "Musique n/a"
    )
    return f"{voice_text} | {music_text}"


def _session_controls_label(*, calibrating: bool) -> str:
    if calibrating:
        return "← maintenir 3s pour quitter   → passer calibration"
    return "← maintenir 3s pour quitter"


def _update_exit_hold_state(
    key: int,
    hold_started_at: Optional[float],
    last_seen_at: Optional[float],
    now: float,
    *,
    hold_seconds: float = 3.0,
    repeat_grace: float = 0.60,
) -> tuple[Optional[float], Optional[float], float, bool]:
    if _cv_key_is(key, "left"):
        if hold_started_at is None or last_seen_at is None or (now - last_seen_at) > repeat_grace:
            hold_started_at = now
        last_seen_at = now
        progress = min(1.0, max(0.0, (now - hold_started_at) / max(hold_seconds, 1e-6)))
        return hold_started_at, last_seen_at, progress, progress >= 1.0

    if (
        hold_started_at is not None
        and last_seen_at is not None
        and (now - last_seen_at) <= repeat_grace
    ):
        progress = min(1.0, max(0.0, (now - hold_started_at) / max(hold_seconds, 1e-6)))
        return hold_started_at, last_seen_at, progress, progress >= 1.0

    return None, None, 0.0, False


def _audio_state_text(enabled: bool, supported: bool, available: bool = True) -> str:
    if not supported:
        return "INDISP"
    if not available:
        return "AUCUNE"
    return "ON" if enabled else "OFF"


def _audio_state_colors(state_text: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if state_text == "ON":
        return (16, 56, 28), (75, 220, 120)
    if state_text in {"INDISP", "AUCUNE"}:
        return (44, 38, 20), (220, 185, 90)
    return (52, 24, 24), (255, 115, 115)


def _draw_audio_panel(
    image: np.ndarray,
    *,
    voice: _VoiceCoach,
    music: _MusicCoach,
    calibrating: bool,
    exit_hold_progress: float = 0.0,
) -> None:
    h, w = image.shape[:2]
    panel_w = min(340, max(280, int(w * 0.24)))
    panel_h = 142
    x2 = w - 22
    y2 = h - 108
    x1 = x2 - panel_w
    y1 = y2 - panel_h

    _blend_rect(
        image,
        x1,
        y1,
        x2,
        y2,
        color=(10, 16, 28),
        alpha=0.72,
        border_color=(56, 82, 112),
        border_thickness=1,
    )

    cv2.putText(
        image,
        "AUDIO",
        (x1 + 16, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (240, 232, 208),
        2,
        cv2.LINE_AA,
    )

    subline = "haut voix   bas musique" if not calibrating else "haut voix   bas musique   droite passer"
    cv2.putText(
        image,
        subline,
        (x1 + 16, y1 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 176, 206),
        1,
        cv2.LINE_AA,
    )

    rows = [
        ("HAUT", "Voix", _audio_state_text(voice.enabled, voice.supported)),
        (
            "BAS",
            "Musique",
            _audio_state_text(music.enabled, music.supported, bool(getattr(music, "_tracks", []))),
        ),
    ]

    row_y = y1 + 58
    for key_label, name, state_text in rows:
        fill, border = _audio_state_colors(state_text)
        _blend_rect(
            image,
            x1 + 12,
            row_y - 18,
            x2 - 12,
            row_y + 18,
            color=(16, 24, 38),
            alpha=0.50,
            border_color=(46, 72, 102),
            border_thickness=1,
        )
        cv2.putText(
            image,
            key_label,
            (x1 + 18, row_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (214, 196, 148),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            name,
            (x1 + 86, row_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (234, 241, 248),
            2,
            cv2.LINE_AA,
        )

        chip_w = 88 if state_text in {"INDISP", "AUCUNE"} else 70
        chip_x1 = x2 - chip_w - 18
        chip_x2 = x2 - 18
        chip_y1 = row_y - 14
        chip_y2 = row_y + 12
        _blend_rect(
            image,
            chip_x1,
            chip_y1,
            chip_x2,
            chip_y2,
            color=fill,
            alpha=0.90,
            border_color=border,
            border_thickness=1,
        )
        state_scale = 0.52 if len(state_text) <= 3 else 0.46
        (tw, th), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, state_scale, 2)
        tx = chip_x1 + max(8, (chip_w - tw) // 2)
        ty = chip_y1 + max(18, (chip_y2 - chip_y1 + th) // 2 - 3)
        cv2.putText(
            image,
            state_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            state_scale,
            (245, 248, 255),
            2,
            cv2.LINE_AA,
        )
        row_y += 34

    footer = _session_controls_label(calibrating=calibrating)
    cv2.putText(
        image,
        footer,
        (x1 + 16, y2 - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (170, 196, 224),
        1,
        cv2.LINE_AA,
    )

    bar_x1 = x1 + 16
    bar_x2 = x2 - 16
    bar_y2 = y2 - 10
    bar_y1 = bar_y2 - 10
    _blend_rect(
        image,
        bar_x1,
        bar_y1,
        bar_x2,
        bar_y2,
        color=(14, 22, 36),
        alpha=0.72,
        border_color=(58, 88, 120),
        border_thickness=1,
    )
    fill_w = int((bar_x2 - bar_x1 - 4) * float(np.clip(exit_hold_progress, 0.0, 1.0)))
    if fill_w > 0:
        _blend_rect(
            image,
            bar_x1 + 2,
            bar_y1 + 2,
            bar_x1 + 2 + fill_w,
            bar_y2 - 2,
            color=(232, 170, 92),
            alpha=0.94,
            border_color=None,
            border_thickness=0,
        )


def _ema(prev: Optional[float], value: Optional[float], alpha: float = 0.35) -> Optional[float]:
    if value is None:
        return prev
    if prev is None:
        return float(value)
    a = float(np.clip(alpha, 0.01, 1.0))
    return float((1.0 - a) * prev + a * float(value))


def _make_rep_tracker(*, stable_frames: int = 3, initial_stage: str = "UP") -> Dict[str, Any]:
    return {
        "stable_frames": max(1, int(stable_frames)),
        "stable_stage": initial_stage,
        "candidate_stage": initial_stage,
        "candidate_count": 0,
        "seen_down": False,
    }


def _rep_tracker_step(
    tracker: Dict[str, Any],
    candidate_stage: str,
    *,
    up_label: str = "UP",
    down_label: str = "DOWN",
) -> tuple[str, bool]:
    cand = str(candidate_stage)
    if cand == tracker["candidate_stage"]:
        tracker["candidate_count"] += 1
    else:
        tracker["candidate_stage"] = cand
        tracker["candidate_count"] = 1

    rep_increment = False
    stable_stage = str(tracker["stable_stage"])
    if tracker["candidate_count"] >= int(tracker["stable_frames"]) and cand != stable_stage:
        tracker["stable_stage"] = cand
        stable_stage = cand
        if stable_stage == down_label:
            tracker["seen_down"] = True
        elif stable_stage == up_label and bool(tracker["seen_down"]):
            rep_increment = True
            tracker["seen_down"] = False

    return str(tracker["stable_stage"]), rep_increment


def _blend_rect(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: tuple[int, int, int],
    alpha: float,
    border_color: Optional[tuple[int, int, int]] = None,
    border_thickness: int = 2,
) -> None:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    overlay = roi.copy()
    overlay[:] = color
    cv2.addWeighted(overlay, float(np.clip(alpha, 0.0, 1.0)), roi, 1.0 - float(np.clip(alpha, 0.0, 1.0)), 0.0, roi)
    if border_color is not None and border_thickness > 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)


def _force_cv_fullscreen(window_name: str, config: Dict[str, Any]) -> None:
    disp_w = int(config.get("display_width", 0) or 0)
    disp_h = int(config.get("display_height", 0) or 0)
    valid_display_size = disp_w >= 1000 and disp_h >= 700

    # Some Linux window managers require a "normal -> fullscreen" sequence.
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except Exception:
        pass
    try:
        cv2.moveWindow(window_name, 0, 0)
    except Exception:
        pass
    if valid_display_size:
        try:
            cv2.resizeWindow(window_name, disp_w, disp_h)
        except Exception:
            pass
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass


_CV_KEY_CODES = {
    "left": {81, 2424832, 65361},
    "up": {82, 2490368, 65362},
    "right": {83, 2555904, 65363},
    "down": {84, 2621440, 65364},
}


def _read_cv_key() -> int:
    try:
        if hasattr(cv2, "waitKeyEx"):
            return int(cv2.waitKeyEx(1))
    except Exception:
        pass
    return int(cv2.waitKey(1))


def _cv_key_is(key: int, direction: str) -> bool:
    if key < 0:
        return False
    direction_key = str(direction).strip().lower()
    valid = _CV_KEY_CODES.get(direction_key, set())
    key_low = key & 0xFF
    return key in valid or key_low in valid


def _draw_status_chip(
    image: np.ndarray,
    text: str,
    *,
    x_right: int = 22,
    y_top: int = 22,
) -> None:
    msg = str(text or "").strip()
    if not msg:
        return
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x2 = image.shape[1] - x_right
    x1 = max(0, x2 - tw - 24)
    y1 = y_top
    y2 = y1 + th + 18
    _blend_rect(
        image,
        x1,
        y1,
        x2,
        y2,
        color=(12, 18, 32),
        alpha=0.55,
        border_color=(110, 200, 255),
        border_thickness=2,
    )
    cv2.putText(
        image,
        msg,
        (x1 + 12, y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 245, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_user_hud(
    image: np.ndarray,
    *,
    reps: int,
    progress_percent: float,
    calibrating: bool,
    calibration_remaining: Optional[float],
    metric_label: str = "POMPES",
    metric_value_text: Optional[str] = None,
) -> None:
    h, w = image.shape[:2]

    card_w = max(180, min(250, int(w * 0.18)))
    card_h = max(110, min(136, int(h * 0.16)))
    card_x = 22
    card_y = 22
    _blend_rect(
        image,
        card_x,
        card_y,
        card_x + card_w,
        card_y + card_h,
        color=(10, 16, 28),
        alpha=0.70,
        border_color=(56, 82, 112),
        border_thickness=1,
    )

    accent_w = int(card_w * 0.36)
    _blend_rect(
        image,
        card_x + 14,
        card_y + 14,
        card_x + 14 + accent_w,
        card_y + 40,
        color=(214, 192, 140),
        alpha=0.18,
        border_color=(214, 192, 140),
        border_thickness=1,
    )
    cv2.putText(
        image,
        metric_label[:16].upper(),
        (card_x + 24, card_y + 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (238, 228, 205),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        metric_value_text if metric_value_text is not None else f"{int(reps)}",
        (card_x + 22, card_y + card_h - 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.2,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    if calibrating and calibration_remaining is not None:
        tag = f"CALIB {max(0.0, calibration_remaining):.1f}s"
        tag_x1 = card_x + int(card_w * 0.48)
        tag_y1 = card_y + card_h - 42
        tag_x2 = card_x + card_w - 16
        tag_y2 = tag_y1 + 24
        _blend_rect(
            image,
            tag_x1,
            tag_y1,
            tag_x2,
            tag_y2,
            color=(50, 36, 18),
            alpha=0.75,
            border_color=(214, 176, 90),
            border_thickness=1,
        )
        cv2.putText(
            image,
            tag,
            (tag_x1 + 10, tag_y2 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 224, 150),
            1,
            cv2.LINE_AA,
        )

    bar_margin = 24
    bar_w = max(260, min(560, w - 2 * bar_margin))
    bar_h = 16
    bar_x = (w - bar_w) // 2
    bar_y = h - 26
    _blend_rect(
        image,
        bar_x,
        bar_y - bar_h,
        bar_x + bar_w,
        bar_y,
        color=(10, 16, 28),
        alpha=0.62,
        border_color=(60, 92, 122),
        border_thickness=1,
    )
    fill = int((bar_w - 6) * float(np.clip(progress_percent, 0.0, 100.0)) / 100.0)
    if fill > 0:
        _blend_rect(
            image,
            bar_x + 3,
            bar_y - bar_h + 3,
            bar_x + 3 + fill,
            bar_y - 3,
            color=(74, 208, 255),
            alpha=0.96,
            border_color=None,
            border_thickness=0,
        )


def _draw_user_coach_line(
    image: np.ndarray,
    text: str,
    *,
    severity: str = "info",
) -> None:
    msg = str(text or "").strip()
    if not msg:
        return
    h, w = image.shape[:2]
    box_w = min(int(w * 0.78), 980)
    x1 = (w - box_w) // 2
    x2 = x1 + box_w
    y2 = h - 42
    y1 = y2 - 48
    color_map = {
        "ok": ((14, 32, 22), (72, 204, 126)),
        "warn": ((42, 30, 15), (240, 184, 86)),
        "error": ((46, 18, 18), (245, 112, 112)),
        "info": ((16, 24, 40), (108, 186, 246)),
    }
    fill, border = color_map.get(severity, color_map["info"])
    _blend_rect(
        image,
        x1,
        y1,
        x2,
        y2,
        color=fill,
        alpha=0.64,
        border_color=border,
        border_thickness=1,
    )
    cv2.rectangle(
        image,
        (x1 + 14, y1 + 12),
        (x1 + 22, y2 - 12),
        border,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        msg[:80],
        (x1 + 40, y2 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (242, 247, 252),
        2,
        cv2.LINE_AA,
    )


def _get_challenge_targets(config: Dict[str, Any], exercise_key: str) -> tuple[bool, int, float]:
    enabled = bool(config.get("challenge_enabled", False))
    target_reps = max(0, int(config.get("challenge_target_reps", 0) or 0))
    target_seconds = max(0.0, float(config.get("challenge_target_seconds", 0.0) or 0.0))
    if str(exercise_key).strip().lower() == "plank":
        target_reps = 0
    enabled = enabled and (target_reps > 0 or target_seconds > 0.0)
    return enabled, target_reps, target_seconds


def _format_challenge_stop_reason(
    *,
    target_reps: int,
    target_seconds: float,
    reps: int = 0,
    elapsed_seconds: float = 0.0,
    best_hold_seconds: float = 0.0,
    success: bool = False,
    exercise_key: str = "",
) -> str:
    exercise = str(exercise_key).strip().lower()
    if exercise == "plank":
        hold_label = f"{best_hold_seconds:.1f}s"
        if success:
            return f"objectif maintien atteint ({hold_label})"
        return f"maintien final {hold_label}"

    if target_reps > 0 and target_seconds > 0.0:
        if success:
            return f"{reps} reps en {elapsed_seconds:.1f}s"
        return f"temps ecoule ({reps}/{target_reps} reps)"
    if target_reps > 0:
        return f"{reps}/{target_reps} reps"
    if target_seconds > 0.0:
        return f"{elapsed_seconds:.1f}s / {target_seconds:.0f}s"
    return ""


def _draw_challenge_chip(
    image: np.ndarray,
    *,
    target_reps: int,
    target_seconds: float,
    current_reps: int = 0,
    current_seconds: float = 0.0,
    exercise_key: str = "",
) -> None:
    if target_reps <= 0 and target_seconds <= 0.0:
        return

    lines = ["DEFI"]
    exercise = str(exercise_key).strip().lower()
    if target_reps > 0:
        lines.append(f"Reps {current_reps}/{target_reps}")
    if target_seconds > 0.0:
        label = "Maintien" if exercise == "plank" else "Temps"
        lines.append(f"{label} {current_seconds:.0f}/{target_seconds:.0f}s")

    line_h = 28
    pad = 14
    width = 270
    height = pad * 2 + line_h * len(lines)
    x2 = image.shape[1] - 20
    x1 = x2 - width
    y1 = 20
    y2 = y1 + height
    _blend_rect(
        image,
        x1,
        y1,
        x2,
        y2,
        color=(18, 26, 44),
        alpha=0.58,
        border_color=(255, 205, 90),
        border_thickness=2,
    )
    for idx, line in enumerate(lines):
        color = (255, 225, 145) if idx == 0 else (235, 245, 255)
        scale = 0.72 if idx == 0 else 0.62
        thickness = 2 if idx == 0 else 1
        cv2.putText(
            image,
            line,
            (x1 + pad, y1 + pad + 18 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _extract_best_person_keypoints(tracks: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if tracks is None:
        return None, None
    keypoints = getattr(tracks, "keypoints", None)
    if keypoints is None:
        return None, None
    data = getattr(keypoints, "data", None)
    if data is None:
        return None, None

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    if data.size == 0 or data.ndim != 3 or data.shape[2] < 2:
        return None, None

    xy = data[:, :, :2]
    if data.shape[2] >= 3:
        conf = data[:, :, 2]
    else:
        conf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)

    idx = int(np.argmax(np.mean(conf, axis=1)))
    return xy[idx], conf[idx]


def _extract_yolo_arm_angles(
    kpts_xy: Optional[np.ndarray], kpts_conf: Optional[np.ndarray], min_conf: float = 0.25
) -> Tuple[Optional[float], Optional[float]]:
    if kpts_xy is None or kpts_conf is None:
        return None, None
    if kpts_xy.shape[0] < 11:
        return None, None

    def _one_arm(shoulder_idx: int, elbow_idx: int, wrist_idx: int) -> Optional[float]:
        if (
            float(kpts_conf[shoulder_idx]) < min_conf
            or float(kpts_conf[elbow_idx]) < min_conf
            or float(kpts_conf[wrist_idx]) < min_conf
        ):
            return None
        return _compute_angle(kpts_xy[shoulder_idx], kpts_xy[elbow_idx], kpts_xy[wrist_idx])

    left = _one_arm(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    right = _one_arm(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    return left, right


def _draw_yolo_full_skeleton(
    image: np.ndarray,
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    *,
    min_conf: float = 0.25,
) -> None:
    if kpts_xy is None or kpts_conf is None:
        return
    if kpts_xy.ndim != 2 or kpts_xy.shape[0] < 17 or kpts_xy.shape[1] < 2:
        return

    h, w = image.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for idx in range(min(17, kpts_xy.shape[0])):
        if float(kpts_conf[idx]) < min_conf:
            continue
        x = int(np.clip(float(kpts_xy[idx][0]), 0.0, float(w - 1)))
        y = int(np.clip(float(kpts_xy[idx][1]), 0.0, float(h - 1)))
        points[idx] = (x, y)

    if not points:
        return

    for a, b in YOLO_COCO_CONNECTIONS:
        if a in points and b in points:
            cv2.line(image, points[a], points[b], (90, 214, 255), 2, cv2.LINE_AA)

    for idx, (x, y) in points.items():
        color = (104, 234, 196) if idx >= 5 else (240, 218, 170)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 6, (28, 36, 48), 1, cv2.LINE_AA)


def _plot_yolo_skeleton_or_fallback(tracks: Any, fallback: np.ndarray) -> np.ndarray:
    if tracks is None:
        return fallback
    try:
        return tracks.plot(conf=False, boxes=False, labels=False, probs=False, kpt_line=True)
    except Exception:
        return fallback


def _ensure_mediapipe_model(model_path: str, model_url: str) -> str:
    # 1) Direct path (absolute or relative to current working dir)
    abs_path = os.path.abspath(model_path)
    if os.path.exists(abs_path):
        return abs_path

    # 2) Relative to this source file directory (works when launched from shortcuts)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_module = os.path.join(module_dir, model_path)
    if os.path.exists(candidate_module):
        return os.path.abspath(candidate_module)

    # 3) Relative to executable directory (PyInstaller one-folder mode)
    exe_dir = os.path.dirname(os.path.abspath(getattr(sys, "executable", "")))
    if exe_dir:
        candidate_exe = os.path.join(exe_dir, model_path)
        if os.path.exists(candidate_exe):
            return os.path.abspath(candidate_exe)

    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with urllib.request.urlopen(model_url, timeout=60) as response, open(abs_path, "wb") as f:
        f.write(response.read())
    return abs_path


def _draw_mediapipe_skeleton(
    image: np.ndarray,
    landmarks: list[Any],
    min_visibility: float = 0.5,
    *,
    line_color: tuple[int, int, int] = (90, 214, 255),
    joint_color: tuple[int, int, int] = (104, 234, 196),
    face_joint_color: tuple[int, int, int] = (240, 218, 170),
    label_text: Optional[str] = None,
    label_anchor: Optional[np.ndarray] = None,
) -> None:
    if not landmarks:
        return
    h, w = image.shape[:2]
    points: dict[int, tuple[int, int]] = {}

    for idx, lm in enumerate(landmarks):
        if lm.x is None or lm.y is None:
            continue
        visibility = 1.0 if lm.visibility is None else float(lm.visibility)
        presence = 1.0 if lm.presence is None else float(lm.presence)
        if visibility < min_visibility or presence < min_visibility:
            continue
        px = int(np.clip(float(lm.x), 0.0, 1.0) * (w - 1))
        py = int(np.clip(float(lm.y), 0.0, 1.0) * (h - 1))
        points[idx] = (px, py)

    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        start = int(conn.start)
        end = int(conn.end)
        if start in points and end in points:
            cv2.line(image, points[start], points[end], line_color, 2, cv2.LINE_AA)

    for idx, (px, py) in points.items():
        color = joint_color if idx >= 11 else face_joint_color
        cv2.circle(image, (px, py), 4, color, -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), 6, (28, 36, 48), 1, cv2.LINE_AA)

    if label_text:
        anchor = label_anchor
        if anchor is None:
            for preferred in (0, MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER):
                if preferred in points:
                    anchor = np.array(
                        [points[preferred][0] / max(1, w - 1), points[preferred][1] / max(1, h - 1)],
                        dtype=np.float32,
                    )
                    break
        if anchor is not None:
            ax = int(np.clip(float(anchor[0]), 0.0, 1.0) * (w - 1))
            ay = int(np.clip(float(anchor[1]), 0.0, 1.0) * (h - 1))
            _blend_rect(
                image,
                ax - 30,
                ay - 38,
                ax + 34,
                ay - 12,
                color=(12, 20, 34),
                alpha=0.76,
                border_color=line_color,
                border_thickness=1,
            )
            cv2.putText(
                image,
                label_text[:8],
                (ax - 20, ay - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (244, 247, 252),
                2,
                cv2.LINE_AA,
            )


def _mediapipe_pose_center(
    landmarks: Optional[list[Any]],
    *,
    min_visibility: float = 0.35,
) -> Optional[np.ndarray]:
    if not landmarks:
        return None
    primary_indices = [MP_LEFT_SHOULDER, MP_RIGHT_SHOULDER, MP_LEFT_HIP, MP_RIGHT_HIP]
    points = [
        _mp_point(landmarks, idx, min_visibility=min_visibility)
        for idx in primary_indices
    ]
    valid = [p for p in points if p is not None]
    if len(valid) >= 2:
        return np.mean(valid, axis=0).astype(np.float32)

    fallback_valid = []
    for idx in range(min(len(landmarks), 33)):
        point = _mp_point(landmarks, idx, min_visibility=min_visibility)
        if point is not None:
            fallback_valid.append(point)
    if fallback_valid:
        return np.mean(fallback_valid, axis=0).astype(np.float32)
    return None


def _mediapipe_pose_scale(
    landmarks: Optional[list[Any]],
    *,
    min_visibility: float = 0.35,
) -> float:
    if not landmarks:
        return 0.18
    ls = _mp_point(landmarks, MP_LEFT_SHOULDER, min_visibility=min_visibility)
    rs = _mp_point(landmarks, MP_RIGHT_SHOULDER, min_visibility=min_visibility)
    lh = _mp_point(landmarks, MP_LEFT_HIP, min_visibility=min_visibility)
    rh = _mp_point(landmarks, MP_RIGHT_HIP, min_visibility=min_visibility)
    spans: List[float] = []
    if ls is not None and rs is not None:
        spans.append(_compute_distance_2d(ls, rs))
    if lh is not None and rh is not None:
        spans.append(_compute_distance_2d(lh, rh))
    if ls is not None and rs is not None and lh is not None and rh is not None:
        shoulder_center = (ls + rs) / 2.0
        hip_center = (lh + rh) / 2.0
        spans.append(_compute_distance_2d(shoulder_center, hip_center) * 1.8)
    if spans:
        return max(0.12, float(np.mean(spans)))
    return 0.18


def _mediapipe_pose_quality(
    landmarks: Optional[list[Any]],
    *,
    min_visibility: float = 0.35,
) -> int:
    if not landmarks:
        return 0
    important = [
        0,
        MP_LEFT_SHOULDER,
        MP_RIGHT_SHOULDER,
        MP_LEFT_ELBOW,
        MP_RIGHT_ELBOW,
        MP_LEFT_WRIST,
        MP_RIGHT_WRIST,
        MP_LEFT_HIP,
        MP_RIGHT_HIP,
    ]
    score = 0
    for idx in important:
        if _mp_point(landmarks, idx, min_visibility=min_visibility) is not None:
            score += 1
    return score


def _extract_mediapipe_pose_candidates(
    pose_landmarks: Optional[list[list[Any]]],
    *,
    min_visibility: float = 0.35,
) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    for landmarks in pose_landmarks or []:
        center = _mediapipe_pose_center(landmarks, min_visibility=min_visibility)
        if center is None:
            continue
        left_angle, right_angle = _extract_mediapipe_arm_angles(
            landmarks,
            min_visibility=min_visibility,
        )
        candidates.append(
            {
                "landmarks": landmarks,
                "center": center,
                "scale": _mediapipe_pose_scale(landmarks, min_visibility=min_visibility),
                "quality": _mediapipe_pose_quality(landmarks, min_visibility=min_visibility),
                "left_angle": left_angle,
                "right_angle": right_angle,
            }
        )
    candidates.sort(key=lambda item: (-int(item["quality"]), float(item["center"][0])))
    return candidates


def _match_pose_candidates_to_players(
    players: Dict[int, Dict[str, Any]],
    candidates: list[Dict[str, Any]],
    *,
    now: float,
    max_stale_seconds: float = 2.0,
) -> tuple[Dict[int, Dict[str, Any]], list[Dict[str, Any]]]:
    active_ids = [
        player_id
        for player_id, player in players.items()
        if now - float(player.get("last_seen", 0.0)) <= max_stale_seconds
        and player.get("center") is not None
    ]

    pairings: list[tuple[float, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        center = candidate["center"]
        scale = max(0.12, float(candidate["scale"]))
        for player_id in active_ids:
            player = players[player_id]
            prev_center = player.get("center")
            if prev_center is None:
                continue
            prev_scale = max(0.12, float(player.get("scale", scale)))
            dist = float(np.linalg.norm(center - prev_center))
            norm_dist = dist / max(scale, prev_scale)
            scale_penalty = abs(scale - prev_scale) / max(scale, prev_scale)
            if norm_dist > 1.8 or scale_penalty > 1.1:
                continue
            pairings.append((norm_dist + scale_penalty * 0.22, candidate_index, player_id))

    pairings.sort(key=lambda item: item[0])
    assigned_players: set[int] = set()
    assigned_candidates: set[int] = set()
    matches: Dict[int, Dict[str, Any]] = {}
    for _, candidate_index, player_id in pairings:
        if candidate_index in assigned_candidates or player_id in assigned_players:
            continue
        assigned_candidates.add(candidate_index)
        assigned_players.add(player_id)
        matches[player_id] = candidates[candidate_index]

    unmatched = [
        candidate
        for idx, candidate in enumerate(candidates)
        if idx not in assigned_candidates
    ]
    return matches, unmatched


def _clear_multiplayer_player_presence(player: Dict[str, Any]) -> None:
    player["visible"] = False
    player["landmarks"] = None
    player["progress"] = 0.0
    player["state"] = "PERDU"


def _make_multiplayer_player(
    player_id: int,
    candidate: Optional[Dict[str, Any]] = None,
    *,
    now: float,
) -> Dict[str, Any]:
    source = candidate or {}
    color = MULTI_PLAYER_COLORS[(player_id - 1) % len(MULTI_PLAYER_COLORS)]
    return {
        "id": int(player_id),
        "label": f"J{int(player_id)}",
        "color": color,
        "reps": 0,
        "state": "UP",
        "progress": 0.0,
        "rep_tracker": _make_rep_tracker(stable_frames=3, initial_stage="UP"),
        "smoothed_avg_angle": None,
        "center": source.get("center"),
        "scale": float(source.get("scale", 0.18)),
        "last_seen": float(now) if candidate is not None else -1e9,
        "last_rep_at": 0.0,
        "visible": candidate is not None,
        "landmarks": source.get("landmarks"),
    }


def _sync_multiplayer_players(
    players: Dict[int, Dict[str, Any]],
    candidates: list[Dict[str, Any]],
    *,
    now: float,
    slot_count: int = 4,
) -> None:
    matches, unmatched = _match_pose_candidates_to_players(players, candidates, now=now)
    matched_ids = set(matches.keys())
    for player_id, candidate in matches.items():
        player = players[player_id]
        player["landmarks"] = candidate["landmarks"]
        player["center"] = candidate["center"]
        player["scale"] = float(candidate["scale"])
        player["last_seen"] = float(now)
        player["visible"] = True

    available_ids = [
        player_id
        for player_id in range(1, max(1, int(slot_count)) + 1)
        if player_id in players
        and player_id not in matched_ids
        and (
            players[player_id].get("center") is None
            or float(players[player_id].get("last_seen", -1e9)) < 0.0
        )
    ]
    for candidate, player_id in zip(unmatched, available_ids):
        player = players[player_id]
        player["landmarks"] = candidate["landmarks"]
        player["center"] = candidate["center"]
        player["scale"] = float(candidate["scale"])
        player["last_seen"] = float(now)
        player["visible"] = True
        player["state"] = "UP" if int(player.get("reps", 0)) <= 0 else str(player.get("state", "UP"))

    for player in players.values():
        if now - float(player.get("last_seen", 0.0)) > 0.45:
            _clear_multiplayer_player_presence(player)


def _draw_multiplayer_pushup_hud(
    image: np.ndarray,
    players: Dict[int, Dict[str, Any]],
    *,
    calibrating: bool,
    calibration_remaining: Optional[float],
) -> None:
    ordered = sorted(
        players.values(),
        key=lambda item: int(item.get("id", 0)),
    )
    h, w = image.shape[:2]
    title_h = 58
    title_w = min(360, max(240, int(w * 0.26)))
    _blend_rect(
        image,
        20,
        20,
        20 + title_w,
        20 + title_h,
        color=(10, 18, 30),
        alpha=0.72,
        border_color=(74, 110, 150),
        border_thickness=1,
    )
    cv2.putText(
        image,
        "POMPES MULTI",
        (36, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (244, 247, 252),
        2,
        cv2.LINE_AA,
    )
    if calibrating and calibration_remaining is not None:
        cv2.putText(
            image,
            f"Calibration {max(0.0, calibration_remaining):.1f}s",
            (36, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 210, 122),
            1,
            cv2.LINE_AA,
        )

    card_w = min(250, max(196, int(w * 0.18)))
    card_h = 86
    x = 20
    y = 92
    max_cards = 4
    for player in ordered[:max_cards]:
        color = tuple(int(c) for c in player.get("color", MULTI_PLAYER_COLORS[0]))
        is_visible = bool(player.get("visible", False))
        fill = (12, 20, 34) if is_visible else (18, 18, 18)
        border = color if is_visible else (86, 92, 102)
        _blend_rect(
            image,
            x,
            y,
            x + card_w,
            y + card_h,
            color=fill,
            alpha=0.76,
            border_color=border,
            border_thickness=2,
        )
        cv2.rectangle(image, (x + 12, y + 14), (x + 24, y + card_h - 14), color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(player.get("label", "J?")),
            (x + 36, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (240, 244, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{int(player.get('reps', 0))}",
            (x + 34, y + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.32,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        state_label = "ACTIF" if is_visible else "HORS CADRE"
        cv2.putText(
            image,
            state_label,
            (x + 114, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color if is_visible else (166, 170, 176),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(player.get("state", "UP")),
            (x + 114, y + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (210, 220, 232),
            2,
            cv2.LINE_AA,
        )
        y += card_h + 12


def _extract_mediapipe_arm_angles(
    landmarks: Optional[list[Any]], min_visibility: float = 0.5
) -> Tuple[Optional[float], Optional[float]]:
    if landmarks is None or len(landmarks) <= MP_RIGHT_WRIST:
        return None, None

    def _is_valid(idx: int) -> bool:
        lm = landmarks[idx]
        if lm.x is None or lm.y is None:
            return False
        visibility = 1.0 if lm.visibility is None else float(lm.visibility)
        presence = 1.0 if lm.presence is None else float(lm.presence)
        return visibility >= min_visibility and presence >= min_visibility

    def _to_np(idx: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([float(lm.x), float(lm.y)], dtype=np.float32)

    left = None
    if _is_valid(MP_LEFT_SHOULDER) and _is_valid(MP_LEFT_ELBOW) and _is_valid(MP_LEFT_WRIST):
        left = _compute_angle(
            _to_np(MP_LEFT_SHOULDER),
            _to_np(MP_LEFT_ELBOW),
            _to_np(MP_LEFT_WRIST),
        )

    right = None
    if _is_valid(MP_RIGHT_SHOULDER) and _is_valid(MP_RIGHT_ELBOW) and _is_valid(MP_RIGHT_WRIST):
        right = _compute_angle(
            _to_np(MP_RIGHT_SHOULDER),
            _to_np(MP_RIGHT_ELBOW),
            _to_np(MP_RIGHT_WRIST),
        )

    return left, right


def _format_arms_line(left_angle: Optional[float], right_angle: Optional[float]) -> str:
    if left_angle is not None and right_angle is not None:
        return f"Bras G/D: {left_angle:.1f}/{right_angle:.1f}"
    if left_angle is not None:
        return f"Bras G/D: {left_angle:.1f}/n.a"
    if right_angle is not None:
        return f"Bras G/D: n.a/{right_angle:.1f}"
    return "Bras G/D: n.a/n.a"


def _setup_mediapipe_live_session(
    config: Dict[str, Any],
    *,
    window_name: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if cv2 is None:
        return None, {
            "status": "error",
            "message": (
                "OpenCV manquant. Installe: pip install opencv-python "
                f"(detail import: {_CV2_IMPORT_ERROR})"
            ),
        }
    if mp is None or PoseLandmarker is None:
        return None, {
            "status": "error",
            "message": (
                "MediaPipe manquant. Installe: pip install mediapipe "
                f"(detail import: {_MEDIAPIPE_IMPORT_ERROR})"
            ),
        }

    camera_index = int(config.get("camera_index", 0))
    show_skeleton = bool(config.get("show_skeleton", True))
    calibration_seconds = float(config.get("calibration_seconds", 2.5))
    min_visibility = float(config.get("mediapipe_visibility_conf", 0.5))
    model_path = str(config.get("mediapipe_model", MEDIAPIPE_DEFAULT_MODEL))
    model_url = str(config.get("mediapipe_model_url", MEDIAPIPE_DEFAULT_MODEL_URL))
    min_det_conf = float(config.get("mediapipe_detection_conf", 0.5))
    min_presence_conf = float(config.get("mediapipe_presence_conf", 0.5))
    min_tracking_conf = float(config.get("mediapipe_tracking_conf", 0.5))
    num_poses = max(1, min(4, int(config.get("mediapipe_num_poses", 1))))
    ui_mode = str(config.get("ui_mode", "developer")).strip().lower()

    try:
        model_asset_path = _ensure_mediapipe_model(model_path, model_url)
    except Exception as exc:
        return None, {
            "status": "error",
            "message": f"Impossible de telecharger le modele MediaPipe ({model_url}): {exc}",
        }

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_det_conf,
            min_pose_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
        )
    except Exception as exc:
        return None, {"status": "error", "message": f"Configuration MediaPipe invalide: {exc}"}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None, {
            "status": "error",
            "message": f"Camera {camera_index} indisponible. Essaie index 0/1 dans Parametres.",
        }

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _force_cv_fullscreen(window_name, config)

    return {
        "cap": cap,
        "options": options,
        "camera_index": camera_index,
        "show_skeleton": show_skeleton,
        "calibration_seconds": calibration_seconds,
        "min_visibility": min_visibility,
        "model_asset_path": model_asset_path,
        "user_hud": ui_mode in {"user", "presentation", "jury"},
        "window_config": config,
    }, None


def _run_pushup_session_yolo(config: Dict[str, Any]) -> Dict[str, Any]:
    if cv2 is None:
        return {
            "status": "error",
            "message": (
                "OpenCV manquant. Installe: pip install opencv-python "
                f"(detail import: {_CV2_IMPORT_ERROR})"
            ),
        }
    if not _ensure_ultralytics():
        return {
            "status": "error",
            "message": (
                "Ultralytics manquant. Installe: pip install ultralytics "
                f"(detail import: {_YOLO_IMPORT_ERROR})"
            ),
        }

    camera_index = int(config.get("camera_index", 0))
    show_skeleton = bool(config.get("show_skeleton", True))
    down_threshold = float(config.get("down_angle_threshold", 95))
    up_threshold = float(config.get("up_angle_threshold", 155))
    calibration_seconds = float(config.get("calibration_seconds", 2.5))
    yolo_model_path = str(config.get("yolo_model", "yolo11n-pose.pt"))
    yolo_conf = float(config.get("yolo_conf", 0.25))
    yolo_imgsz = int(config.get("yolo_imgsz", 640))
    ui_mode = str(config.get("ui_mode", "developer")).strip().lower()
    user_hud = ui_mode in {"user", "presentation", "jury"}

    try:
        gym = solutions.AIGym(
            model=yolo_model_path,
            show=False,
            conf=yolo_conf,
            classes=[0],
            kpts=[6, 8, 10],
            up_angle=up_threshold,
            down_angle=down_threshold,
            show_labels=False,
            verbose=False,
        )
        gym.track_add_args["imgsz"] = yolo_imgsz
    except Exception as exc:
        return {
            "status": "error",
            "message": (
                f"Impossible de charger {yolo_model_path}: {exc}. "
                "Installe ultralytics et verifie que le modele est accessible."
            ),
        }

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "status": "error",
            "message": f"Camera {camera_index} indisponible. Essaie index 0/1 dans Parametres.",
        }

    window_name = "Move On - Pompes (YOLO11 Pose)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _force_cv_fullscreen(window_name, config)

    reps = 0
    state = "UP"
    rep_tracker = _make_rep_tracker(stable_frames=3, initial_stage="UP")
    smoothed_avg_angle: Optional[float] = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "pushup"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    prev_time = time.time()
    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            last_good_frame = False
            exit_reason = "Erreur lecture webcam."
            break

        frame = cv2.flip(frame, 1)
        try:
            gym.process(frame.copy())
        except Exception as exc:
            last_good_frame = False
            exit_reason = f"Inference YOLO11 en echec: {exc}"
            break

        kpts_xy, kpts_conf = _extract_best_person_keypoints(gym.tracks)
        left_angle, right_angle = _extract_yolo_arm_angles(kpts_xy, kpts_conf, min_conf=yolo_conf)
        is_down, is_up, avg_angle = _arm_state_from_angles(
            left_angle, right_angle, down_threshold, up_threshold
        )
        smoothed_avg_angle = _ema(smoothed_avg_angle, avg_angle, alpha=0.35)
        angle_for_state = smoothed_avg_angle if smoothed_avg_angle is not None else avg_angle
        progress = _compute_progress_percent(angle_for_state, down_threshold, up_threshold)

        now = time.time()
        dt = max(now - prev_time, 1e-6)
        fps = 1.0 / dt
        prev_time = now

        if calibrating and not calibration_done:
            elapsed = now - calib_start
            remaining = max(0.0, calibration_seconds - elapsed)
            instruction = f"Calibration: place-toi en position ({remaining:.1f}s) - fleche droite pour passer"
            state_display = "CALIB"
            if elapsed >= calibration_seconds:
                calibration_done = True
                reps = 0
                state = "UP"
        else:
            session_elapsed += dt
            candidate_state = _stage_from_angle(
                angle_for_state, down=down_threshold, up=up_threshold
            )
            state, rep_increment = _rep_tracker_step(rep_tracker, candidate_state)
            if rep_increment:
                reps += 1
                voice.announce_rep(reps)
            instruction = "Place-toi face camera. Descends puis remonte completement."
            state_display = state

        challenge_should_end = False
        if calibration_done and challenge_active:
            if challenge_target_reps > 0 and reps >= challenge_target_reps:
                challenge_success = True
                challenge_should_end = True
            elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                challenge_success = challenge_target_reps <= 0
                challenge_should_end = True
            if challenge_should_end:
                challenge_stop_reason = _format_challenge_stop_reason(
                    target_reps=challenge_target_reps,
                    target_seconds=challenge_target_seconds,
                    reps=reps,
                    elapsed_seconds=session_elapsed,
                    success=challenge_success,
                    exercise_key="pushup",
                )

        pushup_coach = "Cadre toi bien face camera"
        pushup_coach_severity = "info"
        if angle_for_state is None:
            pushup_coach = "Place les epaules, coudes et poignets visibles"
            pushup_coach_severity = "warn"
        elif state_display == "UP":
            if float(angle_for_state) < up_threshold - 8:
                pushup_coach = "Tends davantage les bras en haut"
                pushup_coach_severity = "warn"
            else:
                pushup_coach = "Descends de maniere controlee"
                pushup_coach_severity = "ok"
        elif state_display == "DOWN":
            if float(angle_for_state) > down_threshold + 8:
                pushup_coach = "Descends un peu plus bas"
                pushup_coach_severity = "warn"
            else:
                pushup_coach = "Remonte en gardant le corps gainé"
                pushup_coach_severity = "ok"
        elif state_display == "MID":
            pushup_coach = "Mouvement en cours - reste stable"
            pushup_coach_severity = "info"
        elif state_display == "CALIB":
            pushup_coach = "Calibration de la position"
            pushup_coach_severity = "info"
        voice.update_advice(pushup_coach, severity=pushup_coach_severity)

        display = frame.copy()
        if show_skeleton:
            _draw_yolo_full_skeleton(display, kpts_xy, kpts_conf, min_conf=yolo_conf)
            if kpts_xy is None or kpts_conf is None:
                display = _plot_yolo_skeleton_or_fallback(gym.tracks, display)

        if user_hud:
            _draw_user_hud(
                display,
                reps=reps,
                progress_percent=progress,
                calibrating=calibrating and not calibration_done,
                calibration_remaining=remaining if calibrating and not calibration_done else None,
                metric_label="POMPES",
            )
            _draw_user_coach_line(display, pushup_coach, severity=pushup_coach_severity)
            if challenge_active:
                _draw_challenge_chip(
                    display,
                    target_reps=challenge_target_reps,
                    target_seconds=challenge_target_seconds,
                    current_reps=reps,
                    current_seconds=session_elapsed,
                    exercise_key="pushup",
                )
        else:
            lines = [
                f"Backend: YOLO11",
                f"Reps: {reps}",
                f"Etat: {state_display}",
                (
                    f"Angle moyen: {angle_for_state:.1f}"
                    if angle_for_state is not None
                    else "Angle moyen: n/a"
                ),
                _format_arms_line(left_angle, right_angle),
                f"FPS: {fps:.1f}",
                f"Modele: {yolo_model_path}",
                f"Conseil: {pushup_coach}",
                instruction,
                "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
            ]
            if challenge_active:
                lines.insert(3, f"Defi reps: {reps}/{challenge_target_reps}" if challenge_target_reps > 0 else "Defi reps: OFF")
                lines.insert(
                    4,
                    (
                        f"Defi temps: {session_elapsed:.1f}/{challenge_target_seconds:.0f}s"
                        if challenge_target_seconds > 0.0
                        else "Defi temps: OFF"
                    ),
                )
            _draw_hud_lines(display, lines)
            _draw_progress_bar(display, progress)
        _draw_audio_panel(
            display,
            voice=voice,
            music=music,
            calibrating=calibrating and not calibration_done,
            exit_hold_progress=exit_hold_progress,
        )

        cv2.imshow(window_name, display)
        if not window_force_retry_done:
            _force_cv_fullscreen(window_name, config)
            window_force_retry_done = True
        if challenge_should_end:
            break
        key = _read_cv_key()
        left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
            key,
            left_hold_started_at,
            left_hold_last_seen_at,
            time.time(),
        )
        if should_exit:
            break
        if _cv_key_is(key, "up"):
            voice.toggle()
            continue
        if _cv_key_is(key, "down"):
            music.toggle()
            continue
        if _cv_key_is(key, "right") and calibrating and not calibration_done:
            calibration_done = True
            reps = 0
            state = "UP"

    cap.release()
    cv2.destroyWindow(window_name)
    voice.close()
    music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_pushup_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    if cv2 is None:
        return {
            "status": "error",
            "message": (
                "OpenCV manquant. Installe: pip install opencv-python "
                f"(detail import: {_CV2_IMPORT_ERROR})"
            ),
        }
    if mp is None or PoseLandmarker is None:
        return {
            "status": "error",
            "message": (
                "MediaPipe manquant. Installe: pip install mediapipe "
                f"(detail import: {_MEDIAPIPE_IMPORT_ERROR})"
            ),
        }

    camera_index = int(config.get("camera_index", 0))
    show_skeleton = bool(config.get("show_skeleton", True))
    down_threshold = float(config.get("down_angle_threshold", 95))
    up_threshold = float(config.get("up_angle_threshold", 155))
    calibration_seconds = float(config.get("calibration_seconds", 2.5))
    min_visibility = float(config.get("mediapipe_visibility_conf", 0.5))
    model_path = str(config.get("mediapipe_model", MEDIAPIPE_DEFAULT_MODEL))
    model_url = str(config.get("mediapipe_model_url", MEDIAPIPE_DEFAULT_MODEL_URL))

    min_det_conf = float(config.get("mediapipe_detection_conf", 0.5))
    min_presence_conf = float(config.get("mediapipe_presence_conf", 0.5))
    min_tracking_conf = float(config.get("mediapipe_tracking_conf", 0.5))
    num_poses = max(1, min(4, int(config.get("mediapipe_num_poses", 1))))
    ui_mode = str(config.get("ui_mode", "developer")).strip().lower()
    user_hud = ui_mode in {"user", "presentation", "jury"}

    try:
        model_asset_path = _ensure_mediapipe_model(model_path, model_url)
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Impossible de telecharger le modele MediaPipe ({model_url}): {exc}",
        }

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_det_conf,
            min_pose_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
        )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Configuration MediaPipe invalide: {exc}",
        }

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "status": "error",
            "message": f"Camera {camera_index} indisponible. Essaie index 0/1 dans Parametres.",
        }

    window_name = "Move On - Pompes (MediaPipe Pose)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _force_cv_fullscreen(window_name, config)

    reps = 0
    state = "UP"
    rep_tracker = _make_rep_tracker(stable_frames=3, initial_stage="UP")
    smoothed_avg_angle: Optional[float] = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "pushup"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                left_angle, right_angle = _extract_mediapipe_arm_angles(
                    landmarks, min_visibility=min_visibility
                )
                is_down, is_up, avg_angle = _arm_state_from_angles(
                    left_angle, right_angle, down_threshold, up_threshold
                )
                smoothed_avg_angle = _ema(smoothed_avg_angle, avg_angle, alpha=0.35)
                angle_for_state = smoothed_avg_angle if smoothed_avg_angle is not None else avg_angle
                progress = _compute_progress_percent(angle_for_state, down_threshold, up_threshold)

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    instruction = (
                        f"Calibration: place-toi en position ({remaining:.1f}s) - fleche droite pour passer"
                    )
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                        state = "UP"
                else:
                    session_elapsed += dt
                    candidate_state = _stage_from_angle(
                        angle_for_state, down=down_threshold, up=up_threshold
                    )
                    state, rep_increment = _rep_tracker_step(rep_tracker, candidate_state)
                    if rep_increment:
                        reps += 1
                        voice.announce_rep(reps)
                    instruction = "Place-toi face camera. Descends puis remonte completement."
                    state_display = state

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="pushup",
                        )

                pushup_coach = "Cadre toi bien face camera"
                pushup_coach_severity = "info"
                if angle_for_state is None:
                    pushup_coach = "Place les epaules, coudes et poignets visibles"
                    pushup_coach_severity = "warn"
                elif state_display == "UP":
                    if float(angle_for_state) < up_threshold - 8:
                        pushup_coach = "Tends davantage les bras en haut"
                        pushup_coach_severity = "warn"
                    else:
                        pushup_coach = "Descends de maniere controlee"
                        pushup_coach_severity = "ok"
                elif state_display == "DOWN":
                    if float(angle_for_state) > down_threshold + 8:
                        pushup_coach = "Descends un peu plus bas"
                        pushup_coach_severity = "warn"
                    else:
                        pushup_coach = "Remonte en gardant le corps gainé"
                        pushup_coach_severity = "ok"
                elif state_display == "MID":
                    pushup_coach = "Mouvement en cours - reste stable"
                    pushup_coach_severity = "info"
                elif state_display == "CALIB":
                    pushup_coach = "Calibration de la position"
                    pushup_coach_severity = "info"
                voice.update_advice(pushup_coach, severity=pushup_coach_severity)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(
                        display,
                        landmarks,
                        min_visibility=min(min_visibility, 0.25),
                    )

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=(
                            remaining if calibrating and not calibration_done else None
                        ),
                        metric_label="POMPES",
                    )
                    _draw_user_coach_line(display, pushup_coach, severity=pushup_coach_severity)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="pushup",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe",
                        f"Reps: {reps}",
                        f"Etat: {state_display}",
                        (
                            f"Angle moyen: {angle_for_state:.1f}"
                            if angle_for_state is not None
                            else "Angle moyen: n/a"
                        ),
                        _format_arms_line(left_angle, right_angle),
                        f"FPS: {fps:.1f}",
                        f"Modele: {os.path.basename(model_asset_path)}",
                        f"Conseil: {pushup_coach}",
                        instruction,
                        "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
                    ]
                    if challenge_active:
                        lines.insert(
                            3,
                            (
                                f"Defi reps: {reps}/{challenge_target_reps}"
                                if challenge_target_reps > 0
                                else "Defi reps: OFF"
                            ),
                        )
                        lines.insert(
                            4,
                            (
                                f"Defi temps: {session_elapsed:.1f}/{challenge_target_seconds:.0f}s"
                                if challenge_target_seconds > 0.0
                                else "Defi temps: OFF"
                            ),
                        )
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress)
                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
                    state = "UP"
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_pushup_multi_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    if cv2 is None:
        return {
            "status": "error",
            "message": (
                "OpenCV manquant. Installe: pip install opencv-python "
                f"(detail import: {_CV2_IMPORT_ERROR})"
            ),
        }
    if mp is None or PoseLandmarker is None:
        return {
            "status": "error",
            "message": (
                "MediaPipe manquant. Installe: pip install mediapipe "
                f"(detail import: {_MEDIAPIPE_IMPORT_ERROR})"
            ),
        }

    session_config = dict(config)
    session_config["mediapipe_num_poses"] = 4
    camera_index = int(session_config.get("camera_index", 0))
    show_skeleton = bool(session_config.get("show_skeleton", True))
    calibration_seconds = float(session_config.get("calibration_seconds", 2.5))
    min_visibility = float(session_config.get("mediapipe_visibility_conf", 0.5))
    min_det_conf = min(0.35, float(session_config.get("mediapipe_detection_conf", 0.5)))
    min_presence_conf = min(0.35, float(session_config.get("mediapipe_presence_conf", 0.5)))
    model_path = str(
        session_config.get(
            "mediapipe_multi_model",
            session_config.get("mediapipe_model_full", MEDIAPIPE_FULL_MODEL),
        )
    )
    model_url = str(
        session_config.get(
            "mediapipe_multi_model_url",
            session_config.get("mediapipe_model_full_url", MEDIAPIPE_FULL_MODEL_URL),
        )
    )

    try:
        model_asset_path = _ensure_mediapipe_model(model_path, model_url)
    except Exception:
        fallback_path = str(session_config.get("mediapipe_model", MEDIAPIPE_DEFAULT_MODEL))
        fallback_url = str(session_config.get("mediapipe_model_url", MEDIAPIPE_DEFAULT_MODEL_URL))
        try:
            model_asset_path = _ensure_mediapipe_model(fallback_path, fallback_url)
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Impossible de charger le modele MediaPipe multi: {exc}",
            }

    try:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=min_det_conf,
            min_pose_presence_confidence=min_presence_conf,
            min_tracking_confidence=0.0,
        )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Configuration MediaPipe multi invalide: {exc}",
        }

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {
            "status": "error",
            "message": f"Camera {camera_index} indisponible. Essaie index 0/1 dans Parametres.",
        }

    window_name = "Move On - Pompes Multi (MediaPipe)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    _force_cv_fullscreen(window_name, session_config)

    voice = _make_voice_coach(session_config)
    music = _MusicCoach(session_config)
    players: Dict[int, Dict[str, Any]] = {
        player_id: _make_multiplayer_player(player_id, now=time.time())
        for player_id in range(1, 5)
    }
    session_elapsed = 0.0
    prev_time = time.time()

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0
    window_force_retry_done = False

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                try:
                    result = landmarker.detect(mp_image)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                prev_time = now
                fps = 1.0 / dt

                candidates = _extract_mediapipe_pose_candidates(
                    result.pose_landmarks if result.pose_landmarks else [],
                    min_visibility=max(0.3, min_visibility),
                )
                _sync_multiplayer_players(
                    players,
                    candidates,
                    now=now,
                    slot_count=4,
                )

                remaining: Optional[float] = None
                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        for player in players.values():
                            player["reps"] = 0
                            player["state"] = "UP"
                            player["progress"] = 0.0
                            player["smoothed_avg_angle"] = None
                            player["rep_tracker"] = _make_rep_tracker(
                                stable_frames=3,
                                initial_stage="UP",
                            )
                    coach_text = "Calibration multi-joueurs en cours"
                    coach_severity = "info"
                else:
                    session_elapsed += dt
                    rep_events: list[tuple[str, int]] = []
                    visible_count = 0
                    for player in players.values():
                        landmarks = player.get("landmarks")
                        is_visible = bool(player.get("visible", False))
                        if not is_visible or landmarks is None:
                            player["state"] = "PERDU"
                            player["progress"] = 0.0
                            continue
                        visible_count += 1
                        left_angle, right_angle = _extract_mediapipe_arm_angles(
                            landmarks,
                            min_visibility=min_visibility,
                        )
                        _, _, avg_angle = _arm_state_from_angles(
                            left_angle,
                            right_angle,
                            float(session_config.get("down_angle_threshold", 95)),
                            float(session_config.get("up_angle_threshold", 155)),
                        )
                        player["smoothed_avg_angle"] = _ema(
                            player.get("smoothed_avg_angle"),
                            avg_angle,
                            alpha=0.35,
                        )
                        angle_for_state = (
                            player["smoothed_avg_angle"]
                            if player["smoothed_avg_angle"] is not None
                            else avg_angle
                        )
                        player["progress"] = _compute_progress_percent(
                            angle_for_state,
                            float(session_config.get("down_angle_threshold", 95)),
                            float(session_config.get("up_angle_threshold", 155)),
                        )
                        candidate_state = _stage_from_angle(
                            angle_for_state,
                            down=float(session_config.get("down_angle_threshold", 95)),
                            up=float(session_config.get("up_angle_threshold", 155)),
                        )
                        state, rep_increment = _rep_tracker_step(
                            player["rep_tracker"],
                            candidate_state,
                        )
                        player["state"] = state if angle_for_state is not None else "AJUSTE"
                        if rep_increment:
                            player["reps"] = int(player.get("reps", 0)) + 1
                            player["last_rep_at"] = now
                            rep_events.append((str(player["label"]), int(player["reps"])))

                    if rep_events:
                        label, reps = rep_events[0]
                        voice.say(
                            f"{label}, {reps}",
                            key=f"{label}-{reps}",
                            cooldown=3600.0,
                            min_global_interval=0.15,
                        )

                    if visible_count <= 0:
                        coach_text = "Placez les joueurs enti\u00e8rement dans le cadre"
                        coach_severity = "warn"
                    elif visible_count == 1:
                        coach_text = "Deux joueurs minimum pour le mode multi"
                        coach_severity = "warn"
                    else:
                        coach_text = "Suivi multi-joueurs actif"
                        coach_severity = "ok"
                voice.update_advice(coach_text, severity=coach_severity, stable_frames=14)

                display = frame.copy()
                if show_skeleton:
                    ordered_players = sorted(
                        players.values(),
                        key=lambda item: int(item.get("id", 0)),
                    )
                    for player in ordered_players:
                        if not player.get("visible") or player.get("landmarks") is None:
                            continue
                        color = tuple(int(c) for c in player.get("color", MULTI_PLAYER_COLORS[0]))
                        face_color = tuple(min(255, int(c * 0.78 + 52)) for c in color)
                        _draw_mediapipe_skeleton(
                            display,
                            player["landmarks"],
                            min_visibility=min(min_visibility, 0.25),
                            line_color=color,
                            joint_color=color,
                            face_joint_color=face_color,
                            label_text=str(player.get("label", "J?")),
                            label_anchor=player.get("center"),
                        )

                _draw_multiplayer_pushup_hud(
                    display,
                    players,
                    calibrating=calibrating and not calibration_done,
                    calibration_remaining=remaining,
                )
                _draw_user_coach_line(display, coach_text, severity=coach_severity)
                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, session_config)
                    window_force_retry_done = True

                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    for player in players.values():
                        player["reps"] = 0
                        player["state"] = "UP"
                        player["progress"] = 0.0
                        player["smoothed_avg_angle"] = None
                        player["rep_tracker"] = _make_rep_tracker(
                            stable_frames=3,
                            initial_stage="UP",
                        )
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    ranking = sorted(
        (
            {
                "id": int(player.get("id", 0)),
                "label": str(player.get("label", "J?")),
                "reps": int(player.get("reps", 0)),
            }
            for player in players.values()
        ),
        key=lambda item: (-int(item["reps"]), int(item["id"])),
    )
    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "players": ranking,
            "elapsed_seconds": session_elapsed,
        }
    return {
        "status": "ok",
        "players": ranking,
        "elapsed_seconds": session_elapsed,
    }


def _run_squat_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    ctx, err = _setup_mediapipe_live_session(config, window_name="Move On - Squats (MediaPipe)")
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    model_asset_path = str(ctx["model_asset_path"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Squats (MediaPipe)"

    # Stage/angle thresholds inspired by the source project squat detection workflow.
    down_knee_angle = 105.0
    up_knee_angle = 158.0
    middle_knee_angle = 135.0

    reps = 0
    state = "UP"
    rep_tracker = _make_rep_tracker(stable_frames=4, initial_stage="UP")
    smoothed_knee_angle: Optional[float] = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "squat"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                left_knee_angle = _mp_angle(
                    landmarks, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, min_visibility=min_visibility
                )
                right_knee_angle = _mp_angle(
                    landmarks, MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, min_visibility=min_visibility
                )
                avg_knee_angle_raw = _avg_valid(left_knee_angle, right_knee_angle)
                smoothed_knee_angle = _ema(smoothed_knee_angle, avg_knee_angle_raw, alpha=0.3)
                avg_knee_angle = (
                    smoothed_knee_angle if smoothed_knee_angle is not None else avg_knee_angle_raw
                )
                state_from_angle = _stage_from_angle(
                    avg_knee_angle, down=down_knee_angle, up=up_knee_angle
                )
                progress = _progress_from_angle(
                    avg_knee_angle, down=down_knee_angle, up=up_knee_angle
                )

                if state_from_angle == "UP":
                    stage_key = "up"
                elif avg_knee_angle is not None and avg_knee_angle <= middle_knee_angle:
                    stage_key = "down"
                else:
                    stage_key = "middle"
                feet_msg, knees_msg = _squat_placement_feedback(
                    landmarks, stage_key=stage_key, min_visibility=min_visibility
                )

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    instruction = f"Calibration ({remaining:.1f}s) - place-toi de face, pieds visibles"
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                        state = "UP"
                else:
                    session_elapsed += dt
                    state, rep_increment = _rep_tracker_step(rep_tracker, state_from_angle)
                    if rep_increment:
                        reps += 1
                        voice.announce_rep(reps)
                    state_display = state
                    instruction = "Descends en squat puis remonte. Garde pieds et genoux alignes."

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="squat",
                        )

                squat_coach = "Place-toi bien de face"
                squat_severity = "info"
                if avg_knee_angle is None:
                    squat_coach = "Montre tout le corps (epaules, hanches, genoux, pieds)"
                    squat_severity = "warn"
                elif "trop" in feet_msg.lower():
                    if "serres" in feet_msg.lower():
                        squat_coach = "Conseil: ecarte un peu plus les pieds"
                    else:
                        squat_coach = "Conseil: resserre legerement les pieds"
                    squat_severity = "warn"
                elif "trop" in knees_msg.lower():
                    if "serres" in knees_msg.lower():
                        squat_coach = "Conseil: ouvre les genoux dans l'axe des pointes"
                    else:
                        squat_coach = "Conseil: controle l'ouverture des genoux"
                    squat_severity = "warn"
                elif state_display == "DOWN" and avg_knee_angle > down_knee_angle + 8:
                    squat_coach = "Descends un peu plus bas"
                    squat_severity = "warn"
                elif state_display == "UP" and avg_knee_angle < up_knee_angle - 8:
                    squat_coach = "Remonte completement en haut"
                    squat_severity = "warn"
                elif state_display == "MID":
                    squat_coach = "Continue le mouvement, dos gainé"
                    squat_severity = "info"
                elif state_display == "CALIB":
                    squat_coach = "Calibration de la position"
                    squat_severity = "info"
                else:
                    squat_coach = "Squat propre - continue"
                    squat_severity = "ok"
                voice.update_advice(squat_coach, severity=squat_severity)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="SQUATS",
                    )
                    _draw_user_coach_line(display, squat_coach, severity=squat_severity)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="squat",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe",
                        "Exercice: Squat",
                        f"Reps: {reps}",
                        f"Etat: {state_display}",
                        (
                            f"Genoux G/D: {left_knee_angle:.1f}/{right_knee_angle:.1f}"
                            if left_knee_angle is not None and right_knee_angle is not None
                            else "Genoux G/D: n/a"
                        ),
                        f"Angle moyen: {avg_knee_angle:.1f}" if avg_knee_angle is not None else "Angle moyen: n/a",
                        f"Conseil: {squat_coach}",
                        feet_msg,
                        knees_msg,
                        f"FPS: {fps:.1f}  Modele: {os.path.basename(model_asset_path)}",
                        instruction,
                        "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
                    ]
                    if challenge_active:
                        lines.insert(
                            3,
                            (
                                f"Defi reps: {reps}/{challenge_target_reps}"
                                if challenge_target_reps > 0
                                else "Defi reps: OFF"
                            ),
                        )
                        lines.insert(
                            4,
                            (
                                f"Defi temps: {session_elapsed:.1f}/{challenge_target_seconds:.0f}s"
                                if challenge_target_seconds > 0.0
                                else "Defi temps: OFF"
                            ),
                        )
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Amplitude squat")
                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
                    state = "UP"
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_lunge_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    ctx, err = _setup_mediapipe_live_session(config, window_name="Move On - Fentes (MediaPipe)")
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    model_asset_path = str(ctx["model_asset_path"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Fentes (MediaPipe)"

    down_front_knee = 108.0
    up_front_knee = 155.0

    reps = 0
    state = "UP"
    rep_tracker = _make_rep_tracker(stable_frames=4, initial_stage="UP")
    smoothed_front_angle: Optional[float] = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "lunge"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)
    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                left_knee_angle = _mp_angle(
                    landmarks, MP_LEFT_HIP, MP_LEFT_KNEE, MP_LEFT_ANKLE, min_visibility=min_visibility
                )
                right_knee_angle = _mp_angle(
                    landmarks, MP_RIGHT_HIP, MP_RIGHT_KNEE, MP_RIGHT_ANKLE, min_visibility=min_visibility
                )

                front_side = None
                front_angle = None
                if left_knee_angle is not None and right_knee_angle is not None:
                    if left_knee_angle <= right_knee_angle:
                        front_side = "G"
                        front_angle = left_knee_angle
                    else:
                        front_side = "D"
                        front_angle = right_knee_angle
                elif left_knee_angle is not None:
                    front_side = "G"
                    front_angle = left_knee_angle
                elif right_knee_angle is not None:
                    front_side = "D"
                    front_angle = right_knee_angle

                smoothed_front_angle = _ema(smoothed_front_angle, front_angle, alpha=0.28)
                front_angle_for_state = (
                    smoothed_front_angle if smoothed_front_angle is not None else front_angle
                )
                progress = _progress_from_angle(
                    front_angle_for_state, down=down_front_knee, up=up_front_knee
                )
                stage_from_angle = _stage_from_angle(
                    front_angle_for_state, down=down_front_knee, up=up_front_knee
                )

                knee_angle_error = None
                knee_over_toe = None
                if front_side is not None and front_angle is not None:
                    if stage_from_angle == "DOWN":
                        low, high = LUNGE_KNEE_ANGLE_THRESHOLD
                        knee_angle_error = not (low <= front_angle <= high)

                    if front_side == "G":
                        hip = _mp_point(landmarks, MP_LEFT_HIP, min_visibility=min_visibility)
                        knee = _mp_point(landmarks, MP_LEFT_KNEE, min_visibility=min_visibility)
                        toe = _mp_point(landmarks, MP_LEFT_FOOT_INDEX, min_visibility=min_visibility)
                        if toe is None:
                            toe = _mp_point(landmarks, MP_LEFT_ANKLE, min_visibility=min_visibility)
                    else:
                        hip = _mp_point(landmarks, MP_RIGHT_HIP, min_visibility=min_visibility)
                        knee = _mp_point(landmarks, MP_RIGHT_KNEE, min_visibility=min_visibility)
                        toe = _mp_point(landmarks, MP_RIGHT_FOOT_INDEX, min_visibility=min_visibility)
                        if toe is None:
                            toe = _mp_point(landmarks, MP_RIGHT_ANKLE, min_visibility=min_visibility)

                    if hip is not None and knee is not None and toe is not None:
                        direction = float(np.sign(float(toe[0] - hip[0])))
                        if abs(direction) > 0.0 and front_angle < 130.0:
                            knee_over_toe = ((float(knee[0] - toe[0]) * direction) > 0.015)

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    instruction = f"Calibration ({remaining:.1f}s) - place-toi de profil"
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                        state = "UP"
                else:
                    session_elapsed += dt
                    state, rep_increment = _rep_tracker_step(rep_tracker, stage_from_angle)
                    if rep_increment:
                        reps += 1
                        voice.announce_rep(reps)
                    state_display = state
                    instruction = "Avance une jambe, descends, genou avant aligne avec le pied."

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="lunge",
                        )

                lunge_coach = "Place-toi de profil pour une meilleure detection"
                lunge_severity = "info"
                if front_side is None or front_angle_for_state is None:
                    lunge_coach = "Cadre les hanches, genoux et pieds dans l'image"
                    lunge_severity = "warn"
                elif knee_over_toe is True:
                    lunge_coach = "Genou avant trop en avant (depasse la pointe du pied)"
                    lunge_severity = "warn"
                elif knee_angle_error is True and state_display == "DOWN":
                    if front_angle_for_state > LUNGE_KNEE_ANGLE_THRESHOLD[1]:
                        lunge_coach = "Descends un peu plus dans la fente"
                    else:
                        lunge_coach = "Fente trop basse / angle genou trop ferme"
                    lunge_severity = "warn"
                elif state_display == "UP":
                    lunge_coach = "Alterne les jambes, reste droit"
                    lunge_severity = "ok"
                elif state_display == "DOWN":
                    lunge_coach = "Pousse sur le talon avant pour remonter"
                    lunge_severity = "ok"
                elif state_display == "CALIB":
                    lunge_coach = "Calibration de la position"
                    lunge_severity = "info"
                voice.update_advice(lunge_coach, severity=lunge_severity)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="FENTES",
                    )
                    _draw_user_coach_line(display, lunge_coach, severity=lunge_severity)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="lunge",
                        )
                else:
                    warnings: list[str] = []
                    if knee_over_toe is True:
                        warnings.append("Alerte: genou avant depasse la pointe du pied")
                    if knee_angle_error is True:
                        warnings.append("Alerte: angle du genou avant a corriger")
                    if not warnings:
                        warnings.append("Posture: OK / ajustement fin")

                    lines = [
                        "Backend: MediaPipe",
                        "Exercice: Fentes",
                        f"Reps: {reps}",
                        f"Etat: {state_display}",
                        (
                            f"Genou avant ({front_side}): {front_angle_for_state:.1f}"
                            if front_side is not None and front_angle_for_state is not None
                            else "Genou avant: n/a"
                        ),
                        (
                            f"Genoux G/D: {left_knee_angle:.1f}/{right_knee_angle:.1f}"
                            if left_knee_angle is not None and right_knee_angle is not None
                            else "Genoux G/D: n/a"
                        ),
                        f"Conseil: {lunge_coach}",
                        warnings[0],
                        f"FPS: {fps:.1f}  Modele: {os.path.basename(model_asset_path)}",
                        instruction,
                        "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
                    ]
                    if challenge_active:
                        lines.insert(
                            3,
                            (
                                f"Defi reps: {reps}/{challenge_target_reps}"
                                if challenge_target_reps > 0
                                else "Defi reps: OFF"
                            ),
                        )
                        lines.insert(
                            4,
                            (
                                f"Defi temps: {session_elapsed:.1f}/{challenge_target_seconds:.0f}s"
                                if challenge_target_seconds > 0.0
                                else "Defi temps: OFF"
                            ),
                        )
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Amplitude fente")
                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
                    state = "UP"
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_plank_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    ctx, err = _setup_mediapipe_live_session(config, window_name="Move On - Gainage (MediaPipe)")
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    model_asset_path = str(ctx["model_asset_path"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Gainage (MediaPipe)"

    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)
    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    current_hold = 0.0
    best_hold = 0.0
    total_elapsed = 0.0
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    last_hold_callout_bucket = 0
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "plank"
    )
    challenge_success = False
    challenge_stop_reason = ""

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                posture_stage, body_angle, hip_offset, side_used = _plank_posture_metrics(
                    landmarks, min_visibility=min_visibility
                )

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                quality = 0.0
                if body_angle is not None:
                    angle_score = float(np.clip((body_angle - 135.0) / 40.0, 0.0, 1.0))
                    offset_score = 0.0
                    if hip_offset is not None:
                        offset_score = float(np.clip(1.0 - abs(hip_offset) / 0.06, 0.0, 1.0))
                    quality = 100.0 * (0.65 * angle_score + 0.35 * offset_score)

                if posture_stage == "CORRECT":
                    plank_coach = "Gainage correct"
                    plank_severity = "ok"
                elif posture_stage == "HIGH BACK":
                    plank_coach = "Fesses trop relevees"
                    plank_severity = "warn"
                elif posture_stage == "LOW BACK":
                    plank_coach = "Bassin trop bas / dos creuse"
                    plank_severity = "warn"
                elif posture_stage == "ADJUST":
                    plank_coach = "Ajuste l'alignement epaules-hanches-chevilles"
                    plank_severity = "info"
                else:
                    plank_coach = "Position non detectee"
                    plank_severity = "error"
                voice.update_advice(plank_coach, severity=plank_severity, stable_frames=10)

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    instruction = f"Calibration ({remaining:.1f}s) - place-toi de profil pour le gainage"
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        current_hold = 0.0
                        best_hold = 0.0
                        total_elapsed = 0.0
                else:
                    total_elapsed += dt
                    # Keep the timer unless the posture quality collapses to ~0 (user request).
                    if posture_stage != "UNKNOWN" and quality >= 25.0:
                        current_hold += dt
                        best_hold = max(best_hold, current_hold)
                    elif posture_stage != "UNKNOWN" and quality > 0.0:
                        # Freeze timer on weak posture without resetting immediately.
                        current_hold = current_hold
                    else:
                        current_hold = 0.0
                        last_hold_callout_bucket = 0
                    state_display = posture_stage
                    instruction = "Aligne epaules-hanches-chevilles. Corps gainé, regard neutre."

                    bucket = int(current_hold // 5)
                    if bucket >= 1 and bucket > last_hold_callout_bucket:
                        last_hold_callout_bucket = bucket
                        voice.say(
                            f"{bucket * 5} secondes",
                            key=f"plank-sec-{bucket*5}",
                            cooldown=3600.0,
                            min_global_interval=0.3,
                        )

                challenge_should_end = False
                if calibration_done and challenge_active and challenge_target_seconds > 0.0:
                    if current_hold >= challenge_target_seconds:
                        challenge_success = True
                        challenge_should_end = True
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=0,
                            target_seconds=challenge_target_seconds,
                            best_hold_seconds=current_hold,
                            success=True,
                            exercise_key="plank",
                        )

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=0,
                        progress_percent=quality,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="GAINAGE",
                        metric_value_text=f"{current_hold:4.1f}s",
                    )
                    _draw_user_coach_line(display, plank_coach, severity=plank_severity)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_seconds=current_hold,
                            exercise_key="plank",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe",
                        "Exercice: Gainage",
                        f"Etat: {state_display}",
                        f"Conseil: {plank_coach}",
                        f"Maintien actuel: {current_hold:.1f}s",
                        f"Meilleur maintien: {best_hold:.1f}s",
                        f"Qualite posture: {quality:.1f}%",
                        f"Angle corps: {body_angle:.1f}" if body_angle is not None else "Angle corps: n/a",
                        (
                            f"Offset hanche: {hip_offset:+.3f} ({side_used})"
                            if hip_offset is not None and side_used is not None
                            else "Offset hanche: n/a"
                        ),
                        f"FPS: {fps:.1f}  Modele: {os.path.basename(model_asset_path)}",
                        instruction,
                        "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
                    ]
                    if challenge_active:
                        lines.insert(
                            3,
                            (
                                f"Defi maintien: {current_hold:.1f}/{challenge_target_seconds:.0f}s"
                                if challenge_target_seconds > 0.0
                                else "Defi maintien: OFF"
                            ),
                        )
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, quality, label="Qualite gainage")
                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    current_hold = 0.0
                    best_hold = 0.0
                    total_elapsed = 0.0
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "best_hold_seconds": best_hold,
            "elapsed_seconds": total_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "best_hold_seconds": best_hold,
        "elapsed_seconds": total_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_jumping_jacks_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Jumping jacks : bras leves au-dessus des epaules + jambes ecartees."""
    ctx, err = _setup_mediapipe_live_session(config, window_name="Move On - Jumping Jacks (MediaPipe)")
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Jumping Jacks (MediaPipe)"

    reps = 0
    state = "DOWN"  # DOWN = bras bas + pieds serres, UP = bras hauts + pieds ecartes
    rep_tracker = _make_rep_tracker(stable_frames=3, initial_stage="DOWN")
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "jumping_jacks"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    voice.announce_start("Jumping jacks")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                ls = _mp_point(landmarks, MP_LEFT_SHOULDER, min_visibility=min_visibility)
                rs = _mp_point(landmarks, MP_RIGHT_SHOULDER, min_visibility=min_visibility)
                lw = _mp_point(landmarks, MP_LEFT_WRIST, min_visibility=min_visibility)
                rw = _mp_point(landmarks, MP_RIGHT_WRIST, min_visibility=min_visibility)
                la = _mp_point(landmarks, MP_LEFT_ANKLE, min_visibility=min_visibility)
                ra = _mp_point(landmarks, MP_RIGHT_ANKLE, min_visibility=min_visibility)

                arms_up = False
                legs_apart = False
                coach_msg = "Place-toi face a la camera"
                coach_sev = "warn"
                progress = 0.0

                if all(p is not None for p in (ls, rs, lw, rw, la, ra)):
                    shoulder_w = max(_compute_distance_2d(ls, rs), 1e-6)
                    ankle_w = _compute_distance_2d(la, ra)
                    # Bras au-dessus des epaules = Y poignet < Y epaule (image Y descend)
                    arms_up = bool(lw[1] < ls[1] - 0.02 and rw[1] < rs[1] - 0.02)
                    # Jambes ecartees quand largeur chevilles > 1.4 x epaules
                    legs_apart = bool(ankle_w > 1.4 * shoulder_w)

                    # Score de progression : moyenne des deux conditions
                    arm_prog = float(np.clip((ls[1] - lw[1]) * 25.0, 0.0, 1.0))
                    leg_prog = float(np.clip((ankle_w / shoulder_w - 0.9) / 0.7, 0.0, 1.0))
                    progress = 50.0 * arm_prog + 50.0 * leg_prog

                    if arms_up and legs_apart:
                        state_from_detect = "UP"
                    elif not arms_up and not legs_apart:
                        state_from_detect = "DOWN"
                    else:
                        state_from_detect = "MID"

                    if state_from_detect == "UP":
                        coach_msg = "Parfait, redescends !"
                        coach_sev = "ok"
                    elif state_from_detect == "DOWN":
                        coach_msg = "Saute : bras en haut, pieds ecartes"
                        coach_sev = "info"
                    elif arms_up and not legs_apart:
                        coach_msg = "Ecarte les pieds plus large"
                        coach_sev = "warn"
                    elif legs_apart and not arms_up:
                        coach_msg = "Leve bien les bras au-dessus de la tete"
                        coach_sev = "warn"
                else:
                    state_from_detect = "UNKNOWN"
                    coach_msg = "Montre tout le corps (tete, bras, pieds)"
                    coach_sev = "warn"

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                else:
                    session_elapsed += dt
                    state, rep_increment = _rep_tracker_step(rep_tracker, state_from_detect)
                    if rep_increment:
                        reps += 1
                        voice.announce_rep(reps)
                    state_display = state
                    # Encouragement periodique toutes les 25s
                    voice.announce_encouragement(min_interval=25.0)

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="jumping_jacks",
                        )

                voice.update_advice(coach_msg, severity=coach_sev)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="JUMPING JACKS",
                    )
                    _draw_user_coach_line(display, coach_msg, severity=coach_sev)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="jumping_jacks",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe",
                        "Exercice: Jumping Jacks",
                        f"Reps: {reps}",
                        f"Etat: {state_display}",
                        f"Conseil: {coach_msg}",
                        f"FPS: {fps:.1f}",
                        "Maintenir fleche gauche 3s pour quitter  Haut: voix  Bas: musique",
                    ]
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Amplitude jumping jack")

                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
    finally:
        voice.announce_completion(
            reps=reps,
            success=challenge_success,
            challenge_active=challenge_active,
        )
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_seated_shoulder_press_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Inclusif : Shoulder press assis (fauteuil).
    Mouvement : bras flechis au niveau des epaules -> extension au-dessus de la tete.
    Detecte sur l'angle coude et la position du poignet vs epaule.
    """
    ctx, err = _setup_mediapipe_live_session(
        config, window_name="Move On - Developpe assis (Inclusif)"
    )
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Developpe assis (Inclusif)"

    # Angle du coude : ~90 bras replies, ~170 bras tendus au-dessus
    down_elbow_angle = 100.0
    up_elbow_angle = 160.0

    reps = 0
    rep_tracker = _make_rep_tracker(stable_frames=3, initial_stage="DOWN")
    smoothed_elbow: Optional[float] = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "shoulder_press"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    voice.announce_start("Developpe epaules assis")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                left_elbow = _mp_angle(
                    landmarks, MP_LEFT_SHOULDER, MP_LEFT_ELBOW, MP_LEFT_WRIST, min_visibility=min_visibility
                )
                right_elbow = _mp_angle(
                    landmarks, MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW, MP_RIGHT_WRIST, min_visibility=min_visibility
                )
                avg_elbow_raw = _avg_valid(left_elbow, right_elbow)
                smoothed_elbow = _ema(smoothed_elbow, avg_elbow_raw, alpha=0.35)
                avg_elbow = smoothed_elbow if smoothed_elbow is not None else avg_elbow_raw

                # Verifier que les poignets montent au-dessus des epaules en position UP
                ls = _mp_point(landmarks, MP_LEFT_SHOULDER, min_visibility=min_visibility)
                rs = _mp_point(landmarks, MP_RIGHT_SHOULDER, min_visibility=min_visibility)
                lw = _mp_point(landmarks, MP_LEFT_WRIST, min_visibility=min_visibility)
                rw = _mp_point(landmarks, MP_RIGHT_WRIST, min_visibility=min_visibility)

                wrists_above = False
                if all(p is not None for p in (ls, rs, lw, rw)):
                    wrists_above = bool(lw[1] < ls[1] and rw[1] < rs[1])

                state_from_angle = _stage_from_angle(
                    avg_elbow, down=down_elbow_angle, up=up_elbow_angle
                )
                # On n'accepte le "UP" que si les poignets sont au-dessus des epaules
                if state_from_angle == "UP" and not wrists_above:
                    state_from_angle = "MID"

                progress = _progress_from_angle(
                    avg_elbow, down=down_elbow_angle, up=up_elbow_angle
                )

                coach_msg = "Place-toi face a la camera, torse visible"
                coach_sev = "warn"
                if avg_elbow is None:
                    coach_msg = "Montre epaules, coudes et poignets"
                    coach_sev = "warn"
                elif state_from_angle == "UP":
                    coach_msg = "Redescends en controle"
                    coach_sev = "ok"
                elif state_from_angle == "DOWN":
                    coach_msg = "Pousse les mains vers le plafond"
                    coach_sev = "info"
                elif state_from_angle == "MID":
                    if not wrists_above:
                        coach_msg = "Monte les mains au-dessus de la tete"
                        coach_sev = "warn"
                    else:
                        coach_msg = "Continue la poussee"
                        coach_sev = "info"

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                else:
                    session_elapsed += dt
                    _, rep_increment = _rep_tracker_step(rep_tracker, state_from_angle)
                    if rep_increment:
                        reps += 1
                        voice.announce_rep(reps)
                    state_display = state_from_angle
                    voice.announce_encouragement(min_interval=25.0)

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="shoulder_press",
                        )

                voice.update_advice(coach_msg, severity=coach_sev)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="DEVELOPPE",
                    )
                    _draw_user_coach_line(display, coach_msg, severity=coach_sev)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="shoulder_press",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe (Inclusif)",
                        "Exercice: Developpe epaules assis",
                        f"Reps: {reps}",
                        f"Etat: {state_display}",
                        (
                            f"Coudes G/D: {left_elbow:.1f}/{right_elbow:.1f}"
                            if left_elbow is not None and right_elbow is not None
                            else "Coudes G/D: n/a"
                        ),
                        f"Conseil: {coach_msg}",
                        f"FPS: {fps:.1f}",
                        "Maintenir fleche gauche 3s pour quitter",
                    ]
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Extension bras")

                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
    finally:
        voice.announce_completion(
            reps=reps,
            success=challenge_success,
            challenge_active=challenge_active,
        )
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_seated_punches_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Inclusif : Coups de poing alternes assis.
    Compte une rep chaque fois qu'un bras passe de plie a tendu vers l avant.
    """
    ctx, err = _setup_mediapipe_live_session(
        config, window_name="Move On - Coups de poing (Inclusif)"
    )
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Coups de poing (Inclusif)"

    down_elbow = 90.0   # coude replie = poing au corps
    up_elbow = 160.0    # coude tendu = extension complete

    reps = 0
    # Deux trackers independants (gauche / droite) pour compter alternativement
    left_tracker = _make_rep_tracker(stable_frames=2, initial_stage="DOWN")
    right_tracker = _make_rep_tracker(stable_frames=2, initial_stage="DOWN")
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "seated_punches"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0
    last_side = ""  # pour feedback alternance

    voice.announce_start("Coups de poing alternes")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                left_elbow = _mp_angle(
                    landmarks, MP_LEFT_SHOULDER, MP_LEFT_ELBOW, MP_LEFT_WRIST, min_visibility=min_visibility
                )
                right_elbow = _mp_angle(
                    landmarks, MP_RIGHT_SHOULDER, MP_RIGHT_ELBOW, MP_RIGHT_WRIST, min_visibility=min_visibility
                )

                left_state = _stage_from_angle(left_elbow, down=down_elbow, up=up_elbow)
                right_state = _stage_from_angle(right_elbow, down=down_elbow, up=up_elbow)

                # Progression basee sur le bras le plus tendu
                progresses = [
                    _progress_from_angle(a, down=down_elbow, up=up_elbow)
                    for a in (left_elbow, right_elbow) if a is not None
                ]
                progress = max(progresses) if progresses else 0.0

                coach_msg = "Alterne les bras, poing ferme"
                coach_sev = "info"
                if left_elbow is None and right_elbow is None:
                    coach_msg = "Montre tes bras face a la camera"
                    coach_sev = "warn"
                elif left_state == "UP" or right_state == "UP":
                    coach_msg = "Beau coup ! Change de bras"
                    coach_sev = "ok"

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                else:
                    session_elapsed += dt
                    _, left_inc = _rep_tracker_step(left_tracker, left_state)
                    _, right_inc = _rep_tracker_step(right_tracker, right_state)
                    if left_inc:
                        reps += 1
                        last_side = "G"
                        voice.announce_rep(reps)
                    if right_inc:
                        reps += 1
                        last_side = "D"
                        voice.announce_rep(reps)
                    state_display = f"L:{left_state} R:{right_state}"
                    voice.announce_encouragement(min_interval=22.0)

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="seated_punches",
                        )

                voice.update_advice(coach_msg, severity=coach_sev)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="COUPS",
                    )
                    _draw_user_coach_line(display, coach_msg, severity=coach_sev)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="seated_punches",
                        )
                else:
                    lines = [
                        "Backend: MediaPipe (Inclusif)",
                        "Exercice: Coups de poing assis",
                        f"Reps: {reps}  Dernier: {last_side or '-'}",
                        f"Etat: {state_display}",
                        f"Conseil: {coach_msg}",
                        f"FPS: {fps:.1f}",
                        "Maintenir fleche gauche 3s pour quitter",
                    ]
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Extension bras")

                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
    finally:
        voice.announce_completion(
            reps=reps,
            success=challenge_success,
            challenge_active=challenge_active,
        )
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def _run_chair_torso_rotation_session_mediapipe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Inclusif : Rotations du tronc assis. Une rep = rotation gauche + rotation droite.
    Calcule l'angle de rotation des epaules par rapport au bassin.
    """
    ctx, err = _setup_mediapipe_live_session(
        config, window_name="Move On - Rotations du tronc (Inclusif)"
    )
    if err is not None or ctx is None:
        return err or {"status": "error", "message": "Erreur initialisation MediaPipe."}

    cap = ctx["cap"]
    options = ctx["options"]
    show_skeleton = bool(ctx["show_skeleton"])
    calibration_seconds = float(ctx["calibration_seconds"])
    min_visibility = float(ctx["min_visibility"])
    user_hud = bool(ctx["user_hud"])
    window_config = dict(ctx.get("window_config", {}))
    window_name = "Move On - Rotations du tronc (Inclusif)"

    # Seuil : difference X (normalisee par largeur epaules) pour detecter rotation
    rotation_threshold = 0.18

    reps = 0
    # State machine : LEFT -> RIGHT -> LEFT = 1 rep complete
    state = "CENTER"
    last_side_completed = None  # "LEFT" ou "RIGHT"
    pending_side = None
    voice = _make_voice_coach(config)
    music = _MusicCoach(config)
    challenge_active, challenge_target_reps, challenge_target_seconds = _get_challenge_targets(
        config, "torso_rotation"
    )
    session_elapsed = 0.0
    challenge_success = False
    challenge_stop_reason = ""
    prev_time = time.time()
    timestamp_ms = int(time.time() * 1000)

    calibrating = calibration_seconds > 0.0
    calibration_done = not calibrating
    calib_start = time.time()

    last_good_frame = True
    exit_reason = "ok"
    window_force_retry_done = False
    left_hold_started_at: Optional[float] = None
    left_hold_last_seen_at: Optional[float] = None
    exit_hold_progress = 0.0

    voice.announce_start("Rotations du tronc")

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    last_good_frame = False
                    exit_reason = "Erreur lecture webcam."
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms += 33

                try:
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as exc:
                    last_good_frame = False
                    exit_reason = f"Inference MediaPipe en echec: {exc}"
                    break

                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                ls = _mp_point(landmarks, MP_LEFT_SHOULDER, min_visibility=min_visibility)
                rs = _mp_point(landmarks, MP_RIGHT_SHOULDER, min_visibility=min_visibility)
                lh = _mp_point(landmarks, MP_LEFT_HIP, min_visibility=min_visibility)
                rh = _mp_point(landmarks, MP_RIGHT_HIP, min_visibility=min_visibility)

                rotation_ratio = None
                side_detected = "CENTER"
                coach_msg = "Tourne doucement le buste a gauche puis a droite"
                coach_sev = "info"
                progress = 0.0

                if all(p is not None for p in (ls, rs, lh, rh)):
                    shoulder_mid_x = (ls[0] + rs[0]) / 2.0
                    hip_mid_x = (lh[0] + rh[0]) / 2.0
                    hip_width = max(_compute_distance_2d(lh, rh), 1e-6)
                    rotation_ratio = (shoulder_mid_x - hip_mid_x) / hip_width

                    if rotation_ratio <= -rotation_threshold:
                        side_detected = "LEFT"
                    elif rotation_ratio >= rotation_threshold:
                        side_detected = "RIGHT"
                    else:
                        side_detected = "CENTER"

                    progress = float(np.clip(abs(rotation_ratio) / rotation_threshold * 100.0, 0.0, 100.0))

                    if side_detected == "LEFT":
                        coach_msg = "Beau virage a gauche, reviens au centre"
                        coach_sev = "ok"
                    elif side_detected == "RIGHT":
                        coach_msg = "Beau virage a droite, reviens au centre"
                        coach_sev = "ok"
                    elif state == "CENTER":
                        coach_msg = "Tourne le tronc d un cote puis de l autre"
                        coach_sev = "info"
                else:
                    coach_msg = "Montre epaules et hanches a la camera"
                    coach_sev = "warn"

                now = time.time()
                dt = max(now - prev_time, 1e-6)
                fps = 1.0 / dt
                prev_time = now

                if calibrating and not calibration_done:
                    elapsed = now - calib_start
                    remaining = max(0.0, calibration_seconds - elapsed)
                    state_display = "CALIB"
                    if elapsed >= calibration_seconds:
                        calibration_done = True
                        reps = 0
                        state = "CENTER"
                        last_side_completed = None
                        pending_side = None
                else:
                    session_elapsed += dt
                    # Machine d etats : on compte 1 rep pour chaque cote tenu brievement
                    # CENTER -> LEFT (pending_side="LEFT") -> back CENTER = 1 rep (G)
                    # CENTER -> RIGHT (pending_side="RIGHT") -> back CENTER = 1 rep (D)
                    if side_detected in ("LEFT", "RIGHT"):
                        pending_side = side_detected
                        state = side_detected
                    elif side_detected == "CENTER" and pending_side is not None:
                        if pending_side != last_side_completed:
                            reps += 1
                            voice.announce_rep(reps)
                            last_side_completed = pending_side
                        pending_side = None
                        state = "CENTER"
                    state_display = state
                    voice.announce_encouragement(min_interval=25.0)

                challenge_should_end = False
                if calibration_done and challenge_active:
                    if challenge_target_reps > 0 and reps >= challenge_target_reps:
                        challenge_success = True
                        challenge_should_end = True
                    elif challenge_target_seconds > 0.0 and session_elapsed >= challenge_target_seconds:
                        challenge_success = challenge_target_reps <= 0
                        challenge_should_end = True
                    if challenge_should_end:
                        challenge_stop_reason = _format_challenge_stop_reason(
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            reps=reps,
                            elapsed_seconds=session_elapsed,
                            success=challenge_success,
                            exercise_key="torso_rotation",
                        )

                voice.update_advice(coach_msg, severity=coach_sev)

                display = frame.copy()
                if show_skeleton and landmarks is not None:
                    _draw_mediapipe_skeleton(display, landmarks, min_visibility=min(min_visibility, 0.25))

                if user_hud:
                    _draw_user_hud(
                        display,
                        reps=reps,
                        progress_percent=progress,
                        calibrating=calibrating and not calibration_done,
                        calibration_remaining=remaining if calibrating and not calibration_done else None,
                        metric_label="ROTATIONS",
                    )
                    _draw_user_coach_line(display, coach_msg, severity=coach_sev)
                    if challenge_active:
                        _draw_challenge_chip(
                            display,
                            target_reps=challenge_target_reps,
                            target_seconds=challenge_target_seconds,
                            current_reps=reps,
                            current_seconds=session_elapsed,
                            exercise_key="torso_rotation",
                        )
                else:
                    ratio_txt = f"{rotation_ratio:+.2f}" if rotation_ratio is not None else "n/a"
                    lines = [
                        "Backend: MediaPipe (Inclusif)",
                        "Exercice: Rotations du tronc",
                        f"Reps: {reps}",
                        f"Etat: {state_display}  Ratio: {ratio_txt}",
                        f"Conseil: {coach_msg}",
                        f"FPS: {fps:.1f}",
                        "Maintenir fleche gauche 3s pour quitter",
                    ]
                    _draw_hud_lines(display, lines)
                    _draw_progress_bar(display, progress, label="Amplitude rotation")

                _draw_audio_panel(
                    display,
                    voice=voice,
                    music=music,
                    calibrating=calibrating and not calibration_done,
                    exit_hold_progress=exit_hold_progress,
                )

                cv2.imshow(window_name, display)
                if not window_force_retry_done:
                    _force_cv_fullscreen(window_name, window_config)
                    window_force_retry_done = True
                if challenge_should_end:
                    break
                key = _read_cv_key()
                left_hold_started_at, left_hold_last_seen_at, exit_hold_progress, should_exit = _update_exit_hold_state(
                    key,
                    left_hold_started_at,
                    left_hold_last_seen_at,
                    time.time(),
                )
                if should_exit:
                    break
                if _cv_key_is(key, "up"):
                    voice.toggle()
                    continue
                if _cv_key_is(key, "down"):
                    music.toggle()
                    continue
                if _cv_key_is(key, "right") and calibrating and not calibration_done:
                    calibration_done = True
                    reps = 0
    finally:
        voice.announce_completion(
            reps=reps,
            success=challenge_success,
            challenge_active=challenge_active,
        )
        cap.release()
        cv2.destroyWindow(window_name)
        voice.close()
        music.close()

    if not last_good_frame:
        return {
            "status": "error",
            "message": exit_reason,
            "reps": reps,
            "elapsed_seconds": session_elapsed,
            "challenge_active": challenge_active,
            "challenge_success": challenge_success,
            "challenge_stop_reason": challenge_stop_reason,
        }
    return {
        "status": "ok",
        "reps": reps,
        "elapsed_seconds": session_elapsed,
        "challenge_active": challenge_active,
        "challenge_success": challenge_success,
        "challenge_stop_reason": challenge_stop_reason,
    }


def run_exercise_session(
    config: Dict[str, Any],
    *,
    exercise: str = "pushup",
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    exercise_key = str(exercise or "pushup").strip().lower()
    if exercise_key in {"pushup_multi", "pushups_multi", "pompes_multi", "multiplayer_pushup"}:
        return _run_pushup_multi_session_mediapipe(config)
    if exercise_key in {"pushup", "pushups", "pompe", "pompes"}:
        return run_pushup_session(config, backend=backend)
    if exercise_key in {"squat", "squats"}:
        return _run_squat_session_mediapipe(config)
    if exercise_key in {"lunge", "lunges", "fente", "fentes"}:
        return _run_lunge_session_mediapipe(config)
    if exercise_key in {"plank", "gainage"}:
        return _run_plank_session_mediapipe(config)
    if exercise_key in {"jumping_jacks", "jumpingjacks", "jumping", "sauts_ecartes"}:
        return _run_jumping_jacks_session_mediapipe(config)
    if exercise_key in {"shoulder_press", "developpe_assis", "developpe_epaules"}:
        return _run_seated_shoulder_press_session_mediapipe(config)
    if exercise_key in {"seated_punches", "punches", "coups_poing"}:
        return _run_seated_punches_session_mediapipe(config)
    if exercise_key in {"torso_rotation", "rotation_tronc", "rotations_tronc"}:
        return _run_chair_torso_rotation_session_mediapipe(config)
    return {"status": "error", "message": f"Exercice non supporte: {exercise_key}"}


def run_pushup_session(config: Dict[str, Any], backend: Optional[str] = None) -> Dict[str, Any]:
    selected_backend = str(backend or config.get("pose_backend", "yolo11")).strip().lower()
    if selected_backend in {"mediapipe_multi", "multipeople_mediapipe", "mp_multi"}:
        return _run_pushup_multi_session_mediapipe(config)
    if selected_backend in {"mediapipe", "media_pipe", "media-pipe", "mp"}:
        return _run_pushup_session_mediapipe(config)
    return _run_pushup_session_yolo(config)
