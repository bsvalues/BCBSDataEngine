"""
Replit authentication module for BCBS Values.

This module implements OpenID Connect authentication with Replit.
"""
import os
from functools import wraps
from urllib.parse import urlencode

from flask import session, redirect, request, url_for, g, render_template
from flask_dance.consumer import (
    OAuth2ConsumerBlueprint,
    oauth_authorized,
    oauth_error,
)
from werkzeug.local import LocalProxy


def make_replit_blueprint():
    """Create a Flask-Dance blueprint for Replit authentication."""
    try:
        repl_id = os.environ['REPL_ID']
    except KeyError:
        print("WARNING: the REPL_ID environment variable must be set")
        repl_id = "unknown_repl_id"  # For development

    issuer_url = os.environ.get('ISSUER_URL', "https://replit.com/oidc")

    try:
        replit_domains = os.environ['REPLIT_DOMAINS']
    except KeyError:
        print("WARNING: the REPLIT_DOMAINS environment variable must be set")
        replit_domains = "http://localhost:5002"  # For development

    redirect_uri = f"{replit_domains.split(',')[0]}"
    
    replit_bp = OAuth2ConsumerBlueprint(
        "replit_auth",
        __name__,
        client_id=repl_id,
        client_secret=None,
        base_url=issuer_url,
        token_url=issuer_url + "/token",
        token_url_params = {
            "auth": (),
            "include_client_id": True,
        },
        authorization_url=issuer_url + "/auth",
        use_pkce=True,
        code_challenge_method="S256",
        scope=["openid", "profile", "email"],
   )

    @replit_bp.before_app_request
    def set_applocal_session():
        g.flask_dance_replit = replit_bp.session

    @replit_bp.route("/logout")
    def logout():
        id_token = None
        if replit_bp.token and 'id_token' in replit_bp.token:
            id_token = replit_bp.token.get('id_token')

        session.pop("replit_user", None)
        del replit_bp.token

        end_session_endpoint = issuer_url + "/session/end"
        encoded_params = urlencode({
            "client_id":
            repl_id,
            "post_logout_redirect_uri":
            request.url_root,
        })
        logout_url = f"{end_session_endpoint}?{encoded_params}"

        return redirect(logout_url)

    @replit_bp.route("/error")
    def error():
        return render_template("403.html"), 403

    return replit_bp


@oauth_authorized.connect
def logged_in(blueprint, token):
    blueprint.token = token
    next_url = session.pop("next_url", None)
    if next_url is not None:
        return redirect(next_url)


@oauth_error.connect
def handle_error(blueprint, error, error_description=None, error_uri=None):
    return redirect(url_for('replit_auth.error'))


def replit_user():
    """Get the current authenticated Replit user."""
    if not replit.authorized:
        return None
    if session.get("replit_user", None) is None:
        resp = replit.get("/oidc/me")
        session["replit_user"] = resp.json()
    return session["replit_user"]


def require_login(f):
    """Decorator to require login for a route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not replit.authorized:
            session["next_url"] = request.url
            return redirect(url_for('replit_auth.login'))
        return f(*args, **kwargs)
    return decorated_function


replit = LocalProxy(lambda: g.flask_dance_replit)