"""Lightweight guest / admin token auth."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import AppSettings


class Role(str, Enum):
    GUEST = "guest"
    ADMIN = "admin"


@dataclass(frozen=True)
class AuthContext:
    role: Role
    token_hash: str


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def resolve_role(token: str, settings: AppSettings) -> Optional[Role]:
    if not token:
        return None
    if token == settings.admin_token:
        return Role.ADMIN
    if token == settings.access_token:
        return Role.GUEST
    return None


def require_auth(
    settings: AppSettings,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> AuthContext:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    role = resolve_role(x_api_key, settings)
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return AuthContext(role=role, token_hash=hash_token(x_api_key))


def require_admin(auth: AuthContext) -> None:
    if auth.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token required",
        )
