"""Sign-in: a trusted header in dev mode, a verified bearer JWT in oidc mode."""

from __future__ import annotations

import jwt
from fastapi import Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.context import AppContext
from tribunal.db.models import User
from tribunal.errors import Unauthorized


def _email_from_token(request: Request, ctx: AppContext) -> str:
    s = ctx.settings
    value = request.headers.get(s.auth_jwt_header, "")
    if s.auth_jwt_header.lower() == "authorization":
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer":
            token = ""
    else:  # an edge proxy such as Cloudflare Access sends the bare JWT in its own header
        token = value.strip()
    if not token:
        raise Unauthorized("sign in required")
    client = getattr(request.app.state, "jwks_client", None)
    if client is None:  # one client keeps the key cache
        client = request.app.state.jwks_client = jwt.PyJWKClient(s.oidc_jwks_url)
    try:
        key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256", "ES256"], audience=s.oidc_audience,
                            issuer=s.oidc_issuer,
                            options={"require": ["exp", "iss", "aud"]})
    except jwt.PyJWTError:
        raise Unauthorized("invalid token") from None
    if s.oidc_require_email_verified and claims.get("email_verified") is not True:
        raise Unauthorized("email is not verified")
    email = claims.get("email")
    if not isinstance(email, str) or not email:
        raise Unauthorized("token has no email claim")
    return email


def get_or_create_user(session: Session, ctx: AppContext, email: str) -> User:
    email = email.strip().lower()
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, is_admin=email in ctx.settings.admin_emails)
        session.add(user)
        session.flush()
    return user


def authenticate(request: Request, ctx: AppContext, session: Session) -> User:
    if ctx.settings.auth_mode == "dev":
        email = request.headers.get("X-Dev-User", "")
    else:
        email = _email_from_token(request, ctx)
    if not email.strip():
        raise Unauthorized("sign in required")
    user = get_or_create_user(session, ctx, email)
    user.is_admin = user.email in ctx.settings.admin_emails  # follows ADMIN_EMAILS both ways
    return user
