"""
sync.py — Pairing QR avec l'application MoveOn (mobile / web HTML)

Flux :
  1. L'app MoveOn génère un QR → JSON { app, code, userId, apiUrl }
  2. MoveOn_Windows11 scanne le QR (ou saisit le code manuellement)
  3. Ce module appelle POST /sync/claim  → enregistre le PC
  4. À la fin de chaque séance  → POST /sync/session  → données envoyées
  5. Déconnexion → état local réinitialisé
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

DEVICE_NAME    = "MoveOn_Windows11"
DEFAULT_API    = "http://localhost:4000/api/v1/public"


class SyncManager:
    """Gère l'état de pairing et les appels API."""

    def __init__(self) -> None:
        self.paired   : bool                   = False
        self.code     : Optional[str]          = None
        self.api_url  : Optional[str]          = None
        self.user_id  : Optional[str]          = None
        self.profile  : Optional[Dict[str, Any]] = None

    # ── Parsing QR ───────────────────────────────────────────────────────────

    @staticmethod
    def parse_qr_payload(raw: str) -> Optional[Dict[str, Any]]:
        """
        Décode le JSON encodé dans le QR MoveOn.
        Retourne le dict { app, code, userId, apiUrl } ou None si invalide.
        """
        try:
            data = json.loads(raw)
            if (
                isinstance(data, dict)
                and data.get("app") == "MoveOn_Windows11"
                and "code"   in data
                and "apiUrl" in data
            ):
                return data
        except Exception:
            pass
        return None

    # ── API calls ─────────────────────────────────────────────────────────────

    def claim(self, code: str, api_url: str = DEFAULT_API) -> Dict[str, Any]:
        """
        POST /sync/claim — appaire ce PC à l'utilisateur MoveOn.
        Lève une exception en cas d'erreur réseau ou HTTP.
        """
        url     = f"{api_url}/sync/claim"
        payload = json.dumps({
            "code":       code.strip().upper(),
            "deviceName": DEVICE_NAME,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read())

        data = body.get("data", {})
        self.paired   = True
        self.code     = code.strip().upper()
        self.api_url  = api_url
        self.user_id  = data.get("userId", "?")
        self.profile  = data.get("profile") or {}
        return data

    def push_session(self, session_data: Dict[str, Any]) -> bool:
        """
        POST /sync/session — envoie les données d'une séance terminée.
        Retourne True si le serveur a confirmé la sauvegarde.
        """
        if not (self.paired and self.code and self.api_url):
            return False
        url     = f"{self.api_url}/sync/session"
        payload = json.dumps({
            "code":    self.code,
            "session": session_data,
        }).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = json.loads(resp.read())
            return bool(body.get("data", {}).get("saved"))
        except Exception:
            return False

    def disconnect(self) -> None:
        """Réinitialise entièrement le pairing local."""
        self.paired  = False
        self.code    = None
        self.api_url = None
        self.user_id = None
        self.profile = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        """Nom affiché de l'utilisateur connecté."""
        if self.profile and self.profile.get("name"):
            return self.profile["name"]
        return self.user_id or "?"
