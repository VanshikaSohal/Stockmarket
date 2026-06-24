"""
JWT-based authentication for the Portfolio Risk Analyzer API.

Provides token creation, verification, and FastAPI dependency injection
for securing endpoints.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.utils.settings import load_settings

security_scheme = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    secret_key: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload claims (e.g. {"sub": "user_id"}).
        secret_key: HMAC signing key.
        expires_delta: Token lifetime. Defaults to 60 minutes.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str, secret_key: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Args:
        token: JWT string.
        secret_key: HMAC signing key.

    Returns:
        Decoded payload dict.

    Raises:
        HTTPException(401): If token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency: extract and validate the current user from a JWT.

    Usage:
        @router.get("/protected")
        async def protected_route(user=Depends(get_current_user)):
            return {"user": user}

    Args:
        credentials: Bearer token from the Authorization header.

    Returns:
        Dict with user claims from the JWT payload.

    Raises:
        HTTPException(403): If no credentials provided.
        HTTPException(401): If token is invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required",
        )

    settings = load_settings()
    secret_key = settings.api_secret_key
    return decode_access_token(credentials.credentials, secret_key)


def admin_required(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency: require admin role.

    Usage:
        @router.get("/admin-only")
        async def admin_route(user=Depends(admin_required)):
            return {"user": user}
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user
