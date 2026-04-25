import json
import threading
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pygame

from vision import run_exercise_session, detect_available_cameras
from sync import SyncManager


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera_index": 0,
    "show_skeleton": True,
    "voice_enabled": False,
    "music_enabled": False,
    "music_volume": 0.45,
    "down_angle_threshold": 95,
    "up_angle_threshold": 155,
    "calibration_seconds": 2.5,
    "pose_backend": "yolo11",
    "yolo_model": "yolo11n-pose.pt",
    "yolo_conf": 0.25,
    "yolo_imgsz": 640,
    "mediapipe_model": "pose_landmarker_lite.task",
    "mediapipe_model_url": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
    "mediapipe_detection_conf": 0.5,
    "mediapipe_presence_conf": 0.5,
    "mediapipe_tracking_conf": 0.5,
    "mediapipe_visibility_conf": 0.5,
}


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        save_config(config_path, DEFAULT_CONFIG.copy())
        return DEFAULT_CONFIG.copy()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}

    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    save_config(config_path, merged)
    return merged


def save_config(config_path: str, config: Dict[str, Any]) -> None:
    path = Path(config_path)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


class ArcadeUI:
    def __init__(self, screen: pygame.Surface, config_path: str) -> None:
        self.screen = screen
        self.config_path = config_path
        self.config = load_config(config_path)
        self.clock = pygame.time.Clock()
        self.running = True

        self.state = "profile_select"
        self.profile_mode = "user"
        self.profile_index = 0
        self.main_index = 0
        self.exercise_index = 0
        self.backend_index = 0
        self.params_index = 0
        self.challenge_index = 0
        self.challenge_exercise_index = 0
        self.challenge_target_reps = 10
        self.challenge_target_seconds = 0
        self.session_return_state = "main"
        self.pending_exercise_key = "pushup"

        self.dev_main_items = [
            "Voir les exercices",
            "Mode defi",
            "Demarrer une session rapide",
            "Session multi-joueurs",
            "Parametres",
            "Sync MoveOn App",
            "Quitter",
        ]
        self.user_main_items = [
            "Voir les exercices",
            "Mode defi",
            "Parametres",
            "Changer de mode",
            "Sync MoveOn App",
            "Quitter",
        ]
        self.profile_items = [
            "Mode Utilisateur demo",
            "Mode Developpeur (complet)",
            "Quitter",
        ]
        self.exercise_entries = [
            ("Pompes", "pushup"),
            ("Squats", "squat"),
            ("Fentes", "lunge"),
            ("Gainage", "plank"),
            ("Jumping Jacks", "jumping_jacks"),
            ("Developpe epaules (assis)", "shoulder_press"),
            ("Coups de poing (assis)", "seated_punches"),
            ("Rotations du tronc (assis)", "torso_rotation"),
            ("Retour", "back"),
        ]
        self.backend_items = ["YOLO11 Pose", "MediaPipe Pose", "Retour"]
        self.backend_return_state = "exercises"

        self.message = ""
        self.message_frames = 0

        # ── QR Sync ──────────────────────────────────────────────────────────
        self.sync        = SyncManager()
        self.qr_mode     = "scanning"   # scanning | manual | connecting | paired | error
        self.qr_camera   = None         # cv2.VideoCapture ouvert dans l'état qr_sync
        self.qr_error    = ""
        self.qr_input    = ""           # saisie manuelle du code
        self.qr_claiming = False        # verrou pour éviter les double-claims

        # ── Selection camera (avant session MediaPipe) ───────────────────────
        self.camera_list: List[Tuple[int, str]] = []   # [(idx, nom), ...]
        self.camera_select_index = 0
        self.pending_launch_args: Optional[Dict[str, Any]] = None

        pygame.mouse.set_visible(False)
        self._reload_visual_assets()

    def _reload_visual_assets(self) -> None:
        width, height = self.screen.get_size()
        self.background = self._build_background(width, height)
        self.title_font = pygame.font.SysFont("bahnschrift", 72, bold=True)
        self.menu_font = pygame.font.SysFont("bahnschrift", 38, bold=True)
        self.small_font = pygame.font.SysFont("bahnschrift", 24)
        self.hint_font = pygame.font.SysFont("bahnschrift", 22)
        self.hero_font = pygame.font.SysFont("bahnschrift", 58, bold=True)

    def _build_background(self, width: int, height: int) -> pygame.Surface:
        surf = pygame.Surface((width, height))
        top = (9, 12, 22)
        mid = (7, 10, 18)
        bottom = (4, 5, 10)
        for y in range(height):
            t = y / max(1, height - 1)
            if t < 0.55:
                u = t / 0.55
                c0 = top
                c1 = mid
            else:
                u = (t - 0.55) / 0.45
                c0 = mid
                c1 = bottom
            color = (
                int(c0[0] * (1.0 - u) + c1[0] * u),
                int(c0[1] * (1.0 - u) + c1[1] * u),
                int(c0[2] * (1.0 - u) + c1[2] * u),
            )
            pygame.draw.line(surf, color, (0, y), (width, y))
        for y in range(0, height, 140):
            pygame.draw.line(surf, (14, 16, 24), (0, y), (width, y), 1)

        # Fine gold accents for a cleaner "premium" feel.
        pygame.draw.line(surf, (110, 90, 48), (0, 96), (width, 96), 1)
        pygame.draw.line(surf, (60, 52, 32), (0, height - 92), (width, height - 92), 1)
        return surf

    def _draw_animated_background(self) -> None:
        self.screen.blit(self.background, (0, 0))
        w, h = self.screen.get_size()
        layer = pygame.Surface((w, h), pygame.SRCALPHA)
        glows = [
            (int(w * 0.18), int(h * 0.22), 260, (24, 42, 70), 18),
            (int(w * 0.82), int(h * 0.20), 240, (38, 34, 22), 14),
            (int(w * 0.70), int(h * 0.78), 300, (14, 22, 40), 12),
        ]
        for cx, cy, radius, color, alpha in glows:
            pygame.draw.circle(layer, (*color, alpha), (cx, cy), radius)
            pygame.draw.circle(layer, (*color, max(4, alpha // 4)), (cx, cy), radius + 110)

        # Keep the center area cleaner so the panels stay dominant.
        center_mask = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(center_mask, (0, 0, 0, 110), (int(w * 0.16), int(h * 0.18), int(w * 0.68), int(h * 0.66)), border_radius=28)
        layer.blit(center_mask, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

        vignette = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(vignette, (0, 0, 0, 0), (0, 0, w, h))
        border = 28
        pygame.draw.rect(vignette, (0, 0, 0, 85), (0, 0, w, border))
        pygame.draw.rect(vignette, (0, 0, 0, 85), (0, h - border, w, border))
        pygame.draw.rect(vignette, (0, 0, 0, 65), (0, 0, border, h))
        pygame.draw.rect(vignette, (0, 0, 0, 65), (w - border, 0, border, h))
        self.screen.blit(layer, (0, 0))
        self.screen.blit(vignette, (0, 0))

    def _draw_glass_panel(
        self,
        rect: pygame.Rect,
        *,
        fill: Tuple[int, int, int, int] = (8, 16, 34, 170),
        border: Tuple[int, int, int] = (100, 190, 255),
        border_alpha: int = 120,
        radius: int = 20,
    ) -> None:
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, fill, panel.get_rect(), border_radius=radius)
        pygame.draw.rect(panel, (*border, border_alpha), panel.get_rect(), 2, border_radius=radius)
        highlight = pygame.Rect(8, 8, max(0, rect.width - 16), max(0, int(rect.height * 0.25)))
        if highlight.width > 0 and highlight.height > 0:
            pygame.draw.rect(panel, (255, 255, 255, 18), highlight, border_radius=max(8, radius - 4))
        self.screen.blit(panel, rect.topleft)

    def _is_user_mode(self) -> bool:
        return str(self.profile_mode).lower() == "user"

    def _main_items(self) -> List[str]:
        return self.user_main_items if self._is_user_mode() else self.dev_main_items

    def _exercise_items(self) -> List[str]:
        return [label for label, _ in self.exercise_entries]

    def _challenge_entries(self) -> List[Tuple[str, str]]:
        return [(label, key) for label, key in self.exercise_entries if key != "back"]

    def _selected_challenge_exercise(self) -> Tuple[str, str]:
        entries = self._challenge_entries()
        idx = max(0, min(self.challenge_exercise_index, len(entries) - 1))
        return entries[idx]

    def _selected_exercise(self) -> Tuple[str, str]:
        idx = max(0, min(self.exercise_index, len(self.exercise_entries) - 1))
        return self.exercise_entries[idx]

    def _exercise_display_name(self, exercise_key: str) -> str:
        if exercise_key == "pushup_multi":
            return "Pompes multi"
        for label, key in self.exercise_entries:
            if key == exercise_key:
                return label
        return "Exercice"

    def _challenge_target_summary(self, exercise_key: str) -> str:
        reps = 0 if exercise_key == "plank" else max(0, int(self.challenge_target_reps))
        seconds = max(0, int(self.challenge_target_seconds))
        if reps > 0 and seconds > 0:
            return f"{reps} reps en {seconds}s"
        if reps > 0:
            return f"{reps} reps"
        if seconds > 0:
            if exercise_key == "plank":
                return f"maintien {seconds}s"
            return f"{seconds}s"
        return "Aucun objectif"

    def _challenge_param_items(self) -> List[Tuple[str, str]]:
        label, exercise_key = self._selected_challenge_exercise()
        reps_value = "N/A" if exercise_key == "plank" else (
            "OFF" if int(self.challenge_target_reps) <= 0 else str(int(self.challenge_target_reps))
        )
        time_label = "Temps maintien" if exercise_key == "plank" else "Temps limite"
        time_value = (
            "OFF"
            if int(self.challenge_target_seconds) <= 0
            else f"{int(self.challenge_target_seconds)} s"
        )
        return [
            ("Exercice", label),
            ("Objectif reps", reps_value),
            (time_label, time_value),
            ("Lancer le defi", ""),
            ("Retour", ""),
        ]

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
            if self.message_frames > 0:
                self.message_frames -= 1
        self._close_qr_camera()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                # Saisie texte en mode manuel QR
                if self.state == "qr_sync" and self.qr_mode == "manual":
                    self._handle_qr_text_input(event)
                else:
                    self._handle_keydown(event.key)

    def _handle_keydown(self, key: int) -> None:
        # ── QR Sync ──────────────────────────────────────────────────────────
        if self.state == "qr_sync":
            self._handle_qr_key(key)
            return

        if self.state == "profile_select":
            if key == pygame.K_UP:
                self.profile_index = (self.profile_index - 1) % len(self.profile_items)
            elif key == pygame.K_DOWN:
                self.profile_index = (self.profile_index + 1) % len(self.profile_items)
            elif key == pygame.K_RIGHT:
                self._activate_profile()
            elif key == pygame.K_LEFT:
                self.running = False
            return

        if self.state == "main":
            main_items = self._main_items()
            if key == pygame.K_UP:
                self.main_index = (self.main_index - 1) % len(main_items)
            elif key == pygame.K_DOWN:
                self.main_index = (self.main_index + 1) % len(main_items)
            elif key == pygame.K_RIGHT:
                self._activate_main()
            elif key == pygame.K_LEFT:
                self.state = "profile_select"
            return

        if self.state == "exercises":
            exercise_items = self._exercise_items()
            if key == pygame.K_UP:
                self.exercise_index = (self.exercise_index - 1) % len(exercise_items)
            elif key == pygame.K_DOWN:
                self.exercise_index = (self.exercise_index + 1) % len(exercise_items)
            elif key == pygame.K_RIGHT:
                self._activate_exercises()
            elif key == pygame.K_LEFT:
                self.state = "main"
            return

        if self.state == "params":
            if key == pygame.K_UP:
                self.params_index = (self.params_index - 1) % len(self._param_items())
            elif key == pygame.K_DOWN:
                self.params_index = (self.params_index + 1) % len(self._param_items())
            elif key == pygame.K_RIGHT:
                if not self._change_param():
                    self._activate_params()
            elif key == pygame.K_LEFT:
                self.state = "main"
            return

        if self.state == "challenge_setup":
            if key == pygame.K_UP:
                self.challenge_index = (self.challenge_index - 1) % len(self._challenge_param_items())
            elif key == pygame.K_DOWN:
                self.challenge_index = (self.challenge_index + 1) % len(self._challenge_param_items())
            elif key == pygame.K_RIGHT:
                if not self._change_challenge_param():
                    self._activate_challenge()
            elif key == pygame.K_LEFT:
                self.state = "main"
            return

        if self.state == "backend_select":
            if key == pygame.K_UP:
                self.backend_index = (self.backend_index - 1) % len(self.backend_items)
            elif key == pygame.K_DOWN:
                self.backend_index = (self.backend_index + 1) % len(self.backend_items)
            elif key == pygame.K_RIGHT:
                self._activate_backend()
            elif key == pygame.K_LEFT:
                self.state = self.backend_return_state
            return

        if self.state == "camera_select":
            items = self._camera_select_items()
            if key == pygame.K_UP:
                self.camera_select_index = (self.camera_select_index - 1) % len(items)
            elif key == pygame.K_DOWN:
                self.camera_select_index = (self.camera_select_index + 1) % len(items)
            elif key == pygame.K_RIGHT:
                self._activate_camera_select()
            elif key == pygame.K_LEFT:
                self._cancel_camera_select()
            return

    def _activate_profile(self) -> None:
        label = self.profile_items[self.profile_index]
        if "Quitter" in label:
            self.running = False
            return
        self.profile_mode = "user" if "Utilisateur" in label else "developer"
        self.state = "main"
        self.main_index = 0
        self.exercise_index = 0
        self.backend_index = 1 if self._is_user_mode() else self.backend_index
        self.params_index = 0
        self.challenge_index = 0
        if self._is_user_mode():
            self._set_message("Mode utilisateur active (MediaPipe, interface simplifiee).")
        else:
            self._set_message("Mode developpeur active (options completes).")

    def _activate_main(self) -> None:
        label = self._main_items()[self.main_index]
        if self._is_user_mode():
            if label == "Voir les exercices":
                self.state = "exercises"
                self.exercise_index = 0
            elif label == "Mode defi":
                self.state = "challenge_setup"
                self.challenge_index = 0
            elif label.startswith("Demarrer une session"):
                self._prompt_camera_then_launch("pushup", backend="mediapipe", return_state="main")
            elif label == "Parametres":
                self.state = "params"
                self.params_index = 0
            elif label == "Changer de mode":
                self.state = "profile_select"
            elif label == "Sync MoveOn App":
                self._enter_qr_sync()
            elif label == "Quitter":
                self.running = False
            return

        if label == "Voir les exercices":
            self.state = "exercises"
        elif label == "Mode defi":
            self.state = "challenge_setup"
            self.challenge_index = 0
        elif label == "Demarrer une session rapide":
            self._open_backend_select("main")
        elif label == "Session multi-joueurs":
            self._prompt_camera_then_launch("pushup_multi", backend="mediapipe", return_state="main")
        elif label == "Parametres":
            self.state = "params"
            self.params_index = 0
        elif label == "Sync MoveOn App":
            self._enter_qr_sync()
        elif label == "Quitter":
            self.running = False

    def _activate_exercises(self) -> None:
        label, exercise_key = self._selected_exercise()
        if exercise_key == "back":
            self.state = "main"
            return

        if self._is_user_mode():
            # Version demo: MediaPipe uniquement, interface simplifiee.
            self._prompt_camera_then_launch(exercise_key, backend="mediapipe", return_state="exercises")
            return

        # Mode developpeur: choix backend uniquement pour les pompes.
        if exercise_key == "pushup":
            self._open_backend_select("exercises", exercise_key=exercise_key)
        else:
            self._prompt_camera_then_launch(exercise_key, backend="mediapipe", return_state="exercises")

    def _open_backend_select(self, return_state: str, exercise_key: str = "pushup") -> None:
        self.backend_return_state = return_state
        self.pending_exercise_key = exercise_key
        current_backend = str(self.config.get("pose_backend", "yolo11")).lower()
        self.backend_index = 1 if current_backend == "mediapipe" else 0
        self.state = "backend_select"

    def _activate_backend(self) -> None:
        label = self.backend_items[self.backend_index]
        if label == "Retour":
            self.state = self.backend_return_state
            return

        backend = "mediapipe" if "MediaPipe" in label else "yolo11"
        self._prompt_camera_then_launch(
            self.pending_exercise_key,
            backend=backend,
            return_state=self.backend_return_state,
        )

    # ── Selection camera ─────────────────────────────────────────────────────

    def _camera_select_items(self) -> List[str]:
        items = [f"{name} (index {idx})" for idx, name in self.camera_list]
        items.append("Retour")
        return items

    def _prompt_camera_then_launch(
        self,
        exercise_key: str,
        backend: Optional[str],
        return_state: Optional[str],
        session_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Affiche le selecteur de camera si plusieurs cameras sont detectees,
        puis lance l exercice. Si une seule camera, lance directement."""
        selected_backend = str(backend or self.config.get("pose_backend", "yolo11")).lower()
        # Selection uniquement pour MediaPipe (YOLO reste sur camera par defaut)
        if selected_backend != "mediapipe":
            self._launch_exercise(
                exercise_key,
                backend=backend,
                return_state=return_state,
                session_overrides=session_overrides,
            )
            return

        self._draw_overlay_message("Detection des cameras...")
        pygame.display.flip()
        try:
            self.camera_list = detect_available_cameras(max_index=5)
        except Exception as exc:
            self.camera_list = []
            self._set_message(f"Detection cameras echouee: {exc}")

        if len(self.camera_list) <= 1:
            # Une seule (ou zero) camera : on lance direct avec index 0 ou celui detecte
            if self.camera_list:
                self.config["camera_index"] = self.camera_list[0][0]
                self._persist_config()
            self._launch_exercise(
                exercise_key,
                backend=backend,
                return_state=return_state,
                session_overrides=session_overrides,
            )
            return

        # Plusieurs cameras : on stocke les parametres et on passe a l ecran de choix
        self.pending_launch_args = {
            "exercise_key": exercise_key,
            "backend": backend,
            "return_state": return_state,
            "session_overrides": session_overrides,
        }
        # Presel de la camera actuellement configuree si possible
        current_idx = int(self.config.get("camera_index", 0))
        self.camera_select_index = 0
        for i, (idx, _name) in enumerate(self.camera_list):
            if idx == current_idx:
                self.camera_select_index = i
                break
        self.state = "camera_select"

    def _activate_camera_select(self) -> None:
        items = self._camera_select_items()
        if self.camera_select_index >= len(items) - 1:  # "Retour"
            self._cancel_camera_select()
            return

        idx, name = self.camera_list[self.camera_select_index]
        self.config["camera_index"] = int(idx)
        self._persist_config()
        self._set_message(f"Camera selectionnee : {name}")

        args = self.pending_launch_args or {}
        self.pending_launch_args = None
        self._launch_exercise(
            exercise_key=args.get("exercise_key", "pushup"),
            backend=args.get("backend"),
            return_state=args.get("return_state"),
            session_overrides=args.get("session_overrides"),
        )

    def _cancel_camera_select(self) -> None:
        """Retour arriere sans lancer la session."""
        args = self.pending_launch_args or {}
        self.pending_launch_args = None
        return_state = args.get("return_state") or "exercises"
        self.state = return_state if return_state in {"main", "exercises"} else "exercises"

    def _activate_params(self) -> None:
        if self._is_user_mode():
            if self.params_index == 5:
                self.state = "main"
            return

        if self.params_index == 7:
            self.state = "main"

    def _change_challenge_param(self) -> bool:
        _, exercise_key = self._selected_challenge_exercise()
        exercise_count = len(self._challenge_entries())
        if self.challenge_index == 0:
            self.challenge_exercise_index = (self.challenge_exercise_index + 1) % exercise_count
            _, exercise_key = self._selected_challenge_exercise()
            if exercise_key == "plank":
                self.challenge_target_reps = 0
            return True
        if self.challenge_index == 1:
            if exercise_key != "plank":
                reps = int(self.challenge_target_reps)
                reps = 0 if reps >= 100 else reps + 5
                self.challenge_target_reps = reps
                return True
            return False
        if self.challenge_index == 2:
            seconds = int(self.challenge_target_seconds)
            seconds = 0 if seconds >= 600 else seconds + 15
            self.challenge_target_seconds = seconds
            return True
        return False

    def _activate_challenge(self) -> None:
        label, exercise_key = self._selected_challenge_exercise()
        if self.challenge_index == 3:
            target_reps = 0 if exercise_key == "plank" else int(self.challenge_target_reps)
            target_seconds = int(self.challenge_target_seconds)
            if target_reps <= 0 and target_seconds <= 0:
                self._set_message("Choisis au moins un objectif: reps, temps, ou les deux.")
                return
            if exercise_key == "plank" and target_seconds <= 0:
                self._set_message("Pour le gainage, choisis une duree cible.")
                return
            overrides = {
                "challenge_enabled": True,
                "challenge_target_reps": max(0, target_reps),
                "challenge_target_seconds": max(0, target_seconds),
            }
            self._launch_exercise(
                exercise_key,
                backend="mediapipe" if self._is_user_mode() else None,
                return_state="challenge_setup",
                session_overrides=overrides,
            )
            return

        if self.challenge_index == 4:
            self.state = "main"

    def _param_items(self) -> List[Tuple[str, str]]:
        if self._is_user_mode():
            return [
                ("Camera index", str(int(self.config["camera_index"]))),
                ("Squelette complet", "ON" if self.config["show_skeleton"] else "OFF"),
                ("Voix", "ON" if self.config["voice_enabled"] else "OFF"),
                ("Musique sport", "ON" if self.config["music_enabled"] else "OFF"),
                ("Calibration (s)", f"{float(self.config['calibration_seconds']):.1f}"),
                ("Retour", ""),
            ]
        return [
            ("Camera index", str(int(self.config["camera_index"]))),
            ("Afficher squelette", "ON" if self.config["show_skeleton"] else "OFF"),
            ("Voix", "ON" if self.config["voice_enabled"] else "OFF"),
            ("Musique sport", "ON" if self.config["music_enabled"] else "OFF"),
            ("Seuil DOWN", str(int(self.config["down_angle_threshold"]))),
            ("Seuil UP", str(int(self.config["up_angle_threshold"]))),
            ("Calibration (s)", f"{float(self.config['calibration_seconds']):.1f}"),
            ("Retour", ""),
        ]

    def _change_param(self) -> bool:
        if self._is_user_mode():
            if self.params_index == 0:
                cam = int(self.config["camera_index"])
                self.config["camera_index"] = 0 if cam >= 5 else cam + 1
                self._persist_config()
                return True
            elif self.params_index == 1:
                self.config["show_skeleton"] = not bool(self.config["show_skeleton"])
                self._persist_config()
                return True
            elif self.params_index == 2:
                self.config["voice_enabled"] = not bool(self.config["voice_enabled"])
                self._persist_config()
                return True
            elif self.params_index == 3:
                self.config["music_enabled"] = not bool(self.config["music_enabled"])
                self._persist_config()
                return True
            elif self.params_index == 4:
                calibration = float(self.config["calibration_seconds"])
                calibration = 0.0 if calibration >= 5.0 else calibration + 0.5
                self.config["calibration_seconds"] = round(calibration, 1)
                self._persist_config()
                return True
            return False

        if self.params_index == 0:
            cam = int(self.config["camera_index"])
            self.config["camera_index"] = 0 if cam >= 5 else cam + 1
            self._persist_config()
            return True
        elif self.params_index == 1:
            self.config["show_skeleton"] = not bool(self.config["show_skeleton"])
            self._persist_config()
            return True
        elif self.params_index == 2:
            self.config["voice_enabled"] = not bool(self.config["voice_enabled"])
            self._persist_config()
            return True
        elif self.params_index == 3:
            self.config["music_enabled"] = not bool(self.config["music_enabled"])
            self._persist_config()
            return True
        elif self.params_index == 4:
            down = int(self.config["down_angle_threshold"])
            up = int(self.config["up_angle_threshold"])
            down = 50 if down >= up - 15 else down + 2
            self.config["down_angle_threshold"] = down
            self._persist_config()
            return True
        elif self.params_index == 5:
            down = int(self.config["down_angle_threshold"])
            up = int(self.config["up_angle_threshold"])
            up = down + 15 if up >= 175 else up + 2
            self.config["up_angle_threshold"] = up
            self._persist_config()
            return True
        elif self.params_index == 6:
            calibration = float(self.config["calibration_seconds"])
            calibration = 0.0 if calibration >= 5.0 else calibration + 0.5
            self.config["calibration_seconds"] = round(calibration, 1)
            self._persist_config()
            return True
        return False

    def _persist_config(self) -> None:
        save_config(self.config_path, self.config)

    def _launch_exercise(
        self,
        exercise_key: str,
        backend: Optional[str] = None,
        return_state: Optional[str] = None,
        session_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        selected_backend = str(backend or self.config.get("pose_backend", "yolo11")).lower()
        if exercise_key in {"squat", "lunge", "plank"}:
            selected_backend = "mediapipe"
        if selected_backend not in {"yolo11", "mediapipe"}:
            selected_backend = "yolo11"
        self.config["pose_backend"] = selected_backend
        self._persist_config()
        self.session_return_state = return_state or ("main" if self._is_user_mode() else "exercises")

        exercise_label = self._exercise_display_name(exercise_key)
        backend_label = "MediaPipe" if selected_backend == "mediapipe" else "YOLO11"
        self._draw_overlay_message(f"Ouverture webcam - {exercise_label} ({backend_label})...")
        pygame.display.flip()
        session_config = dict(self.config)
        session_config["ui_mode"] = "user" if self._is_user_mode() else "developer"
        info = pygame.display.Info()
        screen_w = int(getattr(info, "current_w", 0) or 0)
        screen_h = int(getattr(info, "current_h", 0) or 0)
        if screen_w <= 0 or screen_h <= 0:
            fallback_w, fallback_h = self.screen.get_size()
            screen_w = int(fallback_w)
            screen_h = int(fallback_h)
        session_config["display_width"] = screen_w
        session_config["display_height"] = screen_h
        if session_overrides:
            session_config.update(session_overrides)
        result = run_exercise_session(session_config, exercise=exercise_key, backend=selected_backend)

        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self._reload_visual_assets()
        pygame.event.clear()

        if result.get("status") == "error":
            self._set_message(result.get("message", "Erreur session."))
        else:
            # ── Push séance vers MoveOn si apparié ───────────────────────
            if self.sync.paired:
                session_payload: Dict[str, Any] = {
                    "exercise":    exercise_key,
                    "label":       exercise_label,
                    "backend":     selected_backend,
                    "reps":        int(result.get("reps", 0)),
                    "elapsed_s":   float(result.get("elapsed_seconds", 0.0)),
                    "plank_hold":  float(result.get("best_hold_seconds", 0.0)),
                    "challenge":   bool(result.get("challenge_active", False)),
                    "success":     bool(result.get("challenge_success", False)),
                    "completedAt": datetime.datetime.utcnow().isoformat() + "Z",
                }
                def _push() -> None:
                    ok = self.sync.push_session(session_payload)
                    if ok:
                        self._set_message(
                            f"Seance envoyee a MoveOn ! {session_payload['reps']} reps synchronisees."
                        )
                threading.Thread(target=_push, daemon=True).start()
            if result.get("challenge_active"):
                challenge_success = bool(result.get("challenge_success"))
                stop_reason = str(result.get("challenge_stop_reason", "")).strip()
                if exercise_key == "plank":
                    best_hold = float(result.get("best_hold_seconds", 0.0))
                    prefix = "Defi reussi" if challenge_success else "Defi non valide"
                    detail = stop_reason or f"maintien {best_hold:.1f}s"
                    self._set_message(
                        f"{prefix} ({backend_label}) - {exercise_label}: {detail}"
                    )
                else:
                    reps = int(result.get("reps", 0))
                    elapsed = float(result.get("elapsed_seconds", 0.0))
                    prefix = "Defi reussi" if challenge_success else "Defi non valide"
                    detail = stop_reason or f"{reps} reps en {elapsed:.1f}s"
                    self._set_message(
                        f"{prefix} ({backend_label}) - {exercise_label}: {detail}"
                    )
            elif exercise_key == "plank":
                best_hold = float(result.get("best_hold_seconds", 0.0))
                self._set_message(
                    f"Session terminee ({backend_label}) - {exercise_label}: meilleur maintien {best_hold:.1f}s"
                )
            elif exercise_key == "pushup_multi":
                players = list(result.get("players") or [])
                if players:
                    podium = "  ".join(
                        f"{entry.get('label', 'J?')} {int(entry.get('reps', 0))}"
                        for entry in players[:3]
                    )
                    self._set_message(
                        f"Session terminee ({backend_label}) - {exercise_label}: {podium}"
                    )
                else:
                    self._set_message(
                        f"Session terminee ({backend_label}) - {exercise_label}"
                    )
            else:
                reps = int(result.get("reps", 0))
                self._set_message(
                    f"Session terminee ({backend_label}) - {exercise_label}: {reps} repetitions"
                )

        self.state = self.session_return_state

    def _launch_pushups(self, backend: Optional[str] = None, return_state: Optional[str] = None) -> None:
        self._launch_exercise("pushup", backend=backend, return_state=return_state)

    def _set_message(self, msg: str) -> None:
        self.message = msg
        self.message_frames = 60 * 4

    def _draw(self) -> None:
        self._draw_animated_background()
        if self.state == "profile_select":
            self._draw_menu(
                "MOVE ON",
                self.profile_items,
                self.profile_index,
                subtitle="Choisis l'experience de demonstration",
            )
        elif self.state == "main":
            subtitle = (
                "Version utilisateur - interface simplifiee (MediaPipe)"
                if self._is_user_mode()
                else "Version developpeur - controles complets"
            )
            self._draw_menu("MOVE ON", self._main_items(), self.main_index, subtitle=subtitle)
        elif self.state == "exercises":
            subtitle = (
                "Mode demo - MediaPipe uniquement"
                if self._is_user_mode()
                else "Choix exercice (backend libre pour les pompes)"
            )
            self._draw_menu("EXERCICES", self._exercise_items(), self.exercise_index, subtitle=subtitle)
        elif self.state == "backend_select":
            self._draw_menu(
                "CHOIX MOTEUR",
                self.backend_items,
                self.backend_index,
                subtitle="Disponible uniquement en mode developpeur",
            )
        elif self.state == "camera_select":
            self._draw_menu(
                "CHOIX CAMERA",
                self._camera_select_items(),
                self.camera_select_index,
                subtitle="Selectionne la camera a utiliser pour cette session",
            )
        elif self.state == "challenge_setup":
            self._draw_challenge_setup()
        elif self.state == "params":
            self._draw_params()
        elif self.state == "qr_sync":
            self._draw_qr_sync()

        if self.state == "qr_sync":
            if self.qr_mode == "scanning":
                hint = "Gauche/Echap: retour  M: saisie manuelle du code"
            elif self.qr_mode == "manual":
                hint = "Tape le code (10 car.)  Entree: valider  Echap: annuler"
            elif self.qr_mode == "paired":
                hint = "ESC/Gauche: fermer (reste connecte)  D: se deconnecter"
            elif self.qr_mode == "connecting":
                hint = "Connexion en cours..."
            else:
                hint = "Appuie sur une touche pour reessayer"
        elif self.state == "profile_select":
            hint = "Haut/Bas: naviguer  Droite: choisir  Gauche: quitter"
        elif self.state == "challenge_setup":
            hint = "Haut/Bas: naviguer  Droite: modifier/valider  Gauche: retour"
        elif self._is_user_mode():
            hint = "Haut/Bas: naviguer  Droite: valider/modifier  Gauche: retour"
        else:
            hint = "Haut/Bas: naviguer  Droite: valider/modifier  Gauche: retour"
        hint_surf = self.hint_font.render(hint, True, (180, 210, 255))
        hint_panel = pygame.Rect(28, self.screen.get_height() - 66, hint_surf.get_width() + 26, 38)
        self._draw_glass_panel(
            hint_panel,
            fill=(8, 14, 28, 180),
            border=(95, 165, 230),
            border_alpha=80,
            radius=14,
        )
        self.screen.blit(hint_surf, (hint_panel.x + 13, hint_panel.y + 8))

        if self.message and self.message_frames > 0:
            msg_rect = pygame.Rect(28, 72, self.screen.get_width() - 56, 62)
            self._draw_glass_panel(
                msg_rect,
                fill=(8, 14, 28, 215),
                border=(255, 210, 90),
                border_alpha=135,
                radius=16,
            )
            msg_surf = self.small_font.render(self.message, True, (255, 240, 140))
            self.screen.blit(msg_surf, (msg_rect.x + 18, msg_rect.y + 18))

    def _draw_menu(
        self, title: str, items: List[str], selected: int, subtitle: Optional[str] = None
    ) -> None:
        title_surf = self.hero_font.render(title, True, (245, 235, 210))
        title_x = (self.screen.get_width() - title_surf.get_width()) // 2
        self.screen.blit(title_surf, (title_x, 96))

        accent = pygame.Surface((title_surf.get_width(), 4), pygame.SRCALPHA)
        accent.fill((0, 210, 255, 180))
        self.screen.blit(accent, (title_x, 164))

        if subtitle:
            subtitle_surf = self.small_font.render(subtitle, True, (160, 198, 244))
            sx = (self.screen.get_width() - subtitle_surf.get_width()) // 2
            self.screen.blit(subtitle_surf, (sx, 182))

        panel_width = min(980, self.screen.get_width() - 120)
        panel_height = max(220, 120 + len(items) * 72)
        panel_x = (self.screen.get_width() - panel_width) // 2
        panel_y = 235
        self._draw_glass_panel(
            pygame.Rect(panel_x, panel_y, panel_width, panel_height),
            fill=(7, 14, 28, 170),
            border=(90, 170, 240),
            border_alpha=95,
            radius=22,
        )

        start_y = panel_y + 38
        spacing = 72
        for idx, item in enumerate(items):
            color = (245, 250, 255) if idx == selected else (145, 182, 230)
            text = f"{item}"
            text_surf = self.menu_font.render(text, True, color)
            x = (self.screen.get_width() - text_surf.get_width()) // 2
            y = start_y + idx * spacing
            if idx == selected:
                box = pygame.Rect(x - 22, y - 10, text_surf.get_width() + 44, text_surf.get_height() + 20)
                self._draw_glass_panel(
                    box,
                    fill=(10, 28, 56, 185),
                    border=(0, 210, 255),
                    border_alpha=165,
                    radius=14,
                )
                marker = self.small_font.render(">", True, (255, 205, 90))
                self.screen.blit(marker, (box.x - 26, box.y + 10))
            self.screen.blit(text_surf, (x, y))

    def _draw_params(self) -> None:
        title_surf = self.title_font.render("PARAMETRES", True, (250, 220, 80))
        title_x = (self.screen.get_width() - title_surf.get_width()) // 2
        self.screen.blit(title_surf, (title_x, 84))

        subtitle = (
            "Reglages essentiels pour la demo"
            if self._is_user_mode()
            else "Reglages complets (camera, seuils, calibration)"
        )
        subtitle_surf = self.small_font.render(subtitle, True, (160, 198, 244))
        subtitle_x = (self.screen.get_width() - subtitle_surf.get_width()) // 2
        self.screen.blit(subtitle_surf, (subtitle_x, 158))

        items = self._param_items()
        panel = pygame.Rect(130, 210, self.screen.get_width() - 260, self.screen.get_height() - 360)
        self._draw_glass_panel(
            panel,
            fill=(7, 14, 28, 170),
            border=(90, 170, 240),
            border_alpha=95,
            radius=22,
        )
        start_y = 245
        spacing = 70
        left_x = 220
        right_x = self.screen.get_width() - 220
        for idx, (name, value) in enumerate(items):
            selected = idx == self.params_index
            color = (255, 255, 255) if selected else (140, 176, 228)
            name_surf = self.menu_font.render(name, True, color)
            val_surf = self.menu_font.render(value, True, color)
            y = start_y + idx * spacing

            if selected:
                box = pygame.Rect(170, y - 10, self.screen.get_width() - 340, name_surf.get_height() + 20)
                self._draw_glass_panel(
                    box,
                    fill=(10, 28, 56, 185),
                    border=(0, 210, 255),
                    border_alpha=165,
                    radius=14,
                )

            self.screen.blit(name_surf, (left_x, y))
            self.screen.blit(val_surf, (right_x - val_surf.get_width(), y))

        info = "Droite modifie ou valide l'element selectionne"
        info_surf = self.small_font.render(info, True, (170, 200, 255))
        info_x = (self.screen.get_width() - info_surf.get_width()) // 2
        self.screen.blit(info_surf, (info_x, self.screen.get_height() - 120))

    def _draw_challenge_setup(self) -> None:
        title_surf = self.title_font.render("MODE DEFI", True, (250, 220, 80))
        title_x = (self.screen.get_width() - title_surf.get_width()) // 2
        self.screen.blit(title_surf, (title_x, 84))

        _, exercise_key = self._selected_challenge_exercise()
        subtitle = (
            "Choisis un exercice, un objectif de repetitions, un temps, ou les deux"
            if exercise_key != "plank"
            else "Choisis une duree cible pour le gainage"
        )
        subtitle_surf = self.small_font.render(subtitle, True, (160, 198, 244))
        subtitle_x = (self.screen.get_width() - subtitle_surf.get_width()) // 2
        self.screen.blit(subtitle_surf, (subtitle_x, 158))

        items = self._challenge_param_items()
        panel = pygame.Rect(130, 210, self.screen.get_width() - 260, self.screen.get_height() - 360)
        self._draw_glass_panel(
            panel,
            fill=(7, 14, 28, 170),
            border=(90, 170, 240),
            border_alpha=95,
            radius=22,
        )
        start_y = 245
        spacing = 70
        left_x = 220
        right_x = self.screen.get_width() - 220
        for idx, (name, value) in enumerate(items):
            selected = idx == self.challenge_index
            color = (255, 255, 255) if selected else (140, 176, 228)
            name_surf = self.menu_font.render(name, True, color)
            val_surf = self.menu_font.render(value, True, color)
            y = start_y + idx * spacing

            if selected:
                box = pygame.Rect(170, y - 10, self.screen.get_width() - 340, name_surf.get_height() + 20)
                self._draw_glass_panel(
                    box,
                    fill=(10, 28, 56, 185),
                    border=(0, 210, 255),
                    border_alpha=165,
                    radius=14,
                )

            self.screen.blit(name_surf, (left_x, y))
            self.screen.blit(val_surf, (right_x - val_surf.get_width(), y))

        summary = self._challenge_target_summary(exercise_key)
        summary_surf = self.small_font.render(f"Defi courant: {summary}", True, (255, 220, 120))
        summary_x = (self.screen.get_width() - summary_surf.get_width()) // 2
        self.screen.blit(summary_surf, (summary_x, self.screen.get_height() - 150))

        info = "Droite regle ou lance le defi, gauche revient en arriere"
        info_surf = self.small_font.render(info, True, (170, 200, 255))
        info_x = (self.screen.get_width() - info_surf.get_width()) // 2
        self.screen.blit(info_surf, (info_x, self.screen.get_height() - 115))

    # ═══════════════════════════════════════════════════════════════════════════
    # QR SYNC — Méthodes
    # ═══════════════════════════════════════════════════════════════════════════

    def _enter_qr_sync(self) -> None:
        """Entre dans l'état QR sync et ouvre la caméra."""
        self.qr_mode     = "scanning"
        self.qr_error    = ""
        self.qr_input    = ""
        self.qr_claiming = False
        self.state       = "qr_sync"
        cam_idx = int(self.config.get("camera_index", 0))
        try:
            self.qr_camera = cv2.VideoCapture(cam_idx)
            if not self.qr_camera.isOpened():
                self.qr_camera = None
        except Exception:
            self.qr_camera = None

    def _close_qr_camera(self) -> None:
        """Libère la caméra si elle est ouverte."""
        if self.qr_camera is not None:
            try:
                self.qr_camera.release()
            except Exception:
                pass
            self.qr_camera = None

    def _handle_qr_key(self, key: int) -> None:
        """Gestion clavier dans l'état qr_sync (hors saisie manuelle)."""
        if self.qr_mode == "scanning":
            if key in (pygame.K_ESCAPE, pygame.K_LEFT):
                self._close_qr_camera()
                self.state = "main"
            elif key == pygame.K_m:
                self.qr_mode  = "manual"
                self.qr_input = ""

        elif self.qr_mode == "error":
            self.qr_mode  = "scanning"
            self.qr_error = ""

        elif self.qr_mode == "connecting":
            pass  # attendre la réponse du thread

        elif self.qr_mode == "paired":
            if key == pygame.K_d:
                self.sync.disconnect()
                self._set_message("Deconnecte de MoveOn.")
                self._close_qr_camera()
                self.state = "main"
            elif key in (pygame.K_ESCAPE, pygame.K_LEFT):
                self._close_qr_camera()
                self.state = "main"

    def _handle_qr_text_input(self, event: pygame.event.Event) -> None:
        """Gestion saisie clavier en mode manuel."""
        if event.key == pygame.K_ESCAPE:
            self.qr_mode  = "scanning"
            self.qr_input = ""
        elif event.key == pygame.K_RETURN:
            self._try_claim(self.qr_input.strip().upper(),
                            self.sync.DEFAULT_API if hasattr(self.sync, "DEFAULT_API")
                            else "http://localhost:4000/api/v1/public")
        elif event.key == pygame.K_BACKSPACE:
            self.qr_input = self.qr_input[:-1]
        else:
            ch = event.unicode
            if ch and ch.isprintable() and len(self.qr_input) < 12:
                self.qr_input += ch.upper()

    def _try_claim(self, code: str, api_url: str) -> None:
        """Lance le claim en thread pour ne pas bloquer l'UI."""
        if self.qr_claiming or len(code) < 4:
            if len(code) < 4:
                self.qr_error = "Code trop court (min 4 caracteres)."
                self.qr_mode  = "error"
            return
        self.qr_claiming = True
        self.qr_mode     = "connecting"

        def _worker() -> None:
            try:
                self.sync.claim(code, api_url)
                self.qr_mode     = "paired"
                self.qr_claiming = False
            except Exception as exc:
                self.qr_error    = str(exc)[:80]
                self.qr_mode     = "error"
                self.qr_claiming = False

        threading.Thread(target=_worker, daemon=True).start()

    def _draw_qr_sync(self) -> None:
        """Affiche l'écran de scan QR / pairing."""
        w, h = self.screen.get_size()
        cx   = w // 2

        # ── Titre ────────────────────────────────────────────────────────────
        title_text = "SYNC MOVEON APP"
        if self.sync.paired:
            title_text = "CONNECTE A MOVEON"
        title_surf = self.title_font.render(title_text, True, (80, 210, 255))
        self.screen.blit(title_surf, ((w - title_surf.get_width()) // 2, 70))

        accent = pygame.Surface((title_surf.get_width(), 3), pygame.SRCALPHA)
        accent.fill((80, 210, 255, 180))
        self.screen.blit(accent, ((w - title_surf.get_width()) // 2, 70 + title_surf.get_height() + 4))

        # ── Corps selon le mode ───────────────────────────────────────────────
        body_top = 160

        if self.qr_mode == "scanning":
            self._draw_qr_scanning(body_top, w, h, cx)

        elif self.qr_mode == "manual":
            self._draw_qr_manual(body_top, w, h)

        elif self.qr_mode == "connecting":
            msg = self.small_font.render("Connexion en cours...", True, (200, 220, 255))
            self.screen.blit(msg, (cx - msg.get_width() // 2, h // 2))

        elif self.qr_mode == "paired":
            self._draw_qr_paired(body_top, w, h, cx)

        elif self.qr_mode == "error":
            self._draw_qr_error(body_top, w, h, cx)

    def _draw_qr_scanning(self, top: int, w: int, h: int, cx: int) -> None:
        """Affiche la caméra + instructions de scan."""
        cam_w, cam_h = int(w * 0.45), int(h * 0.52)
        cam_x        = int(w * 0.04)
        cam_y        = top + 10

        frame_drawn = False
        if self.qr_camera is not None and self.qr_camera.isOpened():
            ret, frame = self.qr_camera.read()
            if ret:
                # Détection QR
                if not self.qr_claiming:
                    detector = cv2.QRCodeDetector()
                    data, _, _ = detector.detectAndDecode(frame)
                    if data:
                        payload = SyncManager.parse_qr_payload(data)
                        if payload:
                            self._try_claim(payload["code"], payload["apiUrl"])

                # Affichage frame
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    surf      = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    surf      = pygame.transform.scale(surf, (cam_w, cam_h))
                    # Cadre
                    panel_rect = pygame.Rect(cam_x - 4, cam_y - 4, cam_w + 8, cam_h + 8)
                    self._draw_glass_panel(panel_rect, fill=(0, 0, 0, 200),
                                           border=(0, 210, 255), border_alpha=160, radius=12)
                    self.screen.blit(surf, (cam_x, cam_y))
                    # Viseur vert au centre
                    vis_size = min(cam_w, cam_h) // 3
                    vis_x    = cam_x + (cam_w - vis_size) // 2
                    vis_y    = cam_y + (cam_h - vis_size) // 2
                    pygame.draw.rect(self.screen, (0, 255, 120), (vis_x, vis_y, vis_size, vis_size), 3, border_radius=8)
                    frame_drawn = True
                except Exception:
                    pass

        if not frame_drawn:
            ph_rect = pygame.Rect(cam_x, cam_y, cam_w, cam_h)
            self._draw_glass_panel(ph_rect, fill=(8, 16, 34, 180),
                                   border=(80, 120, 200), border_alpha=100, radius=12)
            no_cam = self.small_font.render("Camera non disponible", True, (160, 180, 220))
            self.screen.blit(no_cam, (cam_x + (cam_w - no_cam.get_width()) // 2,
                                      cam_y + cam_h // 2 - 15))

        # ── Instructions (droite) ────────────────────────────────────────────
        info_x = cam_x + cam_w + int(w * 0.06)
        info_w = w - info_x - 40
        panel  = pygame.Rect(info_x, cam_y, info_w, cam_h)
        self._draw_glass_panel(panel, fill=(7, 14, 28, 170),
                               border=(90, 170, 240), border_alpha=95, radius=18)

        lines = [
            ("1.", "Ouvre l'app MoveOn"),
            ("2.", "Profil → Connecter MoveOn Windows11"),
            ("3.", "Un QR code s'affiche"),
            ("4.", "Pointe la camera vers le QR"),
            ("",   ""),
            ("M",  "Saisie manuelle du code"),
        ]
        ly = cam_y + 28
        for icon, text in lines:
            if icon == "":
                ly += 10
                continue
            col_icon = (255, 210, 80) if icon in ("M",) else (0, 210, 255)
            ic  = self.small_font.render(icon, True, col_icon)
            tx  = self.small_font.render(text, True, (200, 225, 255))
            self.screen.blit(ic, (info_x + 18, ly))
            self.screen.blit(tx, (info_x + 52, ly))
            ly += 38

        # Badge connecté si déjà appairé
        if self.sync.paired:
            badge = self.small_font.render(
                f"Connecte : {self.sync.display_name}", True, (0, 230, 120)
            )
            self.screen.blit(badge, (cx - badge.get_width() // 2, cam_y + cam_h + 20))

    def _draw_qr_manual(self, top: int, w: int, h: int) -> None:
        """Champ de saisie manuelle du code."""
        cx = w // 2
        sub = self.small_font.render(
            "Saisis le code affiche dans l'app MoveOn (Profil > QR)", True, (160, 198, 244)
        )
        self.screen.blit(sub, (cx - sub.get_width() // 2, top))

        # Boîte de saisie
        box_w, box_h = min(600, w - 120), 80
        box = pygame.Rect(cx - box_w // 2, top + 60, box_w, box_h)
        self._draw_glass_panel(box, fill=(10, 22, 48, 200),
                               border=(0, 210, 255), border_alpha=200, radius=16)
        display = self.qr_input if self.qr_input else "..."
        cursor  = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
        inp_surf = self.title_font.render(display + cursor, True, (255, 255, 255))
        self.screen.blit(inp_surf, (cx - inp_surf.get_width() // 2, box.y + (box_h - inp_surf.get_height()) // 2))

        hint2 = self.small_font.render("Entree = valider  •  Echap = retour camera", True, (120, 160, 210))
        self.screen.blit(hint2, (cx - hint2.get_width() // 2, top + 160))

    def _draw_qr_paired(self, top: int, w: int, h: int, cx: int) -> None:
        """Affiche le succès du pairing."""
        panel = pygame.Rect(cx - 340, top, 680, 340)
        self._draw_glass_panel(panel, fill=(4, 28, 12, 200),
                               border=(0, 220, 100), border_alpha=180, radius=22)

        ok = self.hero_font.render("Connexion reussie !", True, (0, 230, 120))
        self.screen.blit(ok, (cx - ok.get_width() // 2, top + 30))

        name_txt = self.menu_font.render(
            f"Utilisateur : {self.sync.display_name}", True, (200, 255, 210)
        )
        self.screen.blit(name_txt, (cx - name_txt.get_width() // 2, top + 115))

        id_txt = self.small_font.render(
            f"ID : {self.sync.user_id}", True, (120, 180, 140)
        )
        self.screen.blit(id_txt, (cx - id_txt.get_width() // 2, top + 168))

        sep = self.small_font.render(
            "Les seances seront automatiquement synchronisees.", True, (160, 220, 180)
        )
        self.screen.blit(sep, (cx - sep.get_width() // 2, top + 215))

        actions = self.small_font.render(
            "ESC / Gauche = fermer   •   D = se deconnecter", True, (180, 200, 255)
        )
        self.screen.blit(actions, (cx - actions.get_width() // 2, top + 260))

    def _draw_qr_error(self, top: int, w: int, h: int, cx: int) -> None:
        """Affiche le message d'erreur."""
        panel = pygame.Rect(cx - 320, top + 20, 640, 200)
        self._draw_glass_panel(panel, fill=(28, 8, 8, 210),
                               border=(220, 60, 60), border_alpha=180, radius=18)
        err_title = self.menu_font.render("Erreur de connexion", True, (255, 100, 80))
        self.screen.blit(err_title, (cx - err_title.get_width() // 2, top + 50))
        err_msg = self.small_font.render(self.qr_error or "Erreur inconnue", True, (255, 180, 160))
        self.screen.blit(err_msg, (cx - err_msg.get_width() // 2, top + 110))
        retry = self.small_font.render("Appuie sur une touche pour reessayer", True, (200, 180, 255))
        self.screen.blit(retry, (cx - retry.get_width() // 2, top + 160))

    def _draw_overlay_message(self, text: str) -> None:
        self._draw_animated_background()
        self._draw_menu("MOVE ON", [], 0, subtitle="Initialisation de la session")
        msg_surf = self.menu_font.render(text, True, (255, 255, 255))
        x = (self.screen.get_width() - msg_surf.get_width()) // 2
        y = (self.screen.get_height() - msg_surf.get_height()) // 2
        panel = pygame.Rect(x - 28, y - 16, msg_surf.get_width() + 56, msg_surf.get_height() + 32)
        self._draw_glass_panel(
            panel,
            fill=(8, 16, 34, 205),
            border=(255, 205, 90),
            border_alpha=145,
            radius=16,
        )
        self.screen.blit(msg_surf, (x, y))
