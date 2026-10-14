"""INFINITY platform: accounts, plans, subscriptions, credit ledger, API keys, audit."""

from __future__ import annotations

from pathlib import Path

from .audit import Audit
from .auth import Auth, AuthError, User
from .billing import Billing, BillingError
from .conversations import ConversationError, Conversations
from .db import PlatformDB
from .keys import Keys, KeyError_


class Platform:
    def __init__(self, db_path: str | Path | None = None, plans_file: Path | None = None, allow_registration: bool = True) -> None:
        self.db = PlatformDB(db_path)
        self.audit = Audit(self.db)
        self.auth = Auth(self.db, self.audit, allow_registration)
        self.billing = Billing(self.db, self.audit, plans_file)
        self.keys = Keys(self.db, self.audit)
        self.conversations = Conversations(self.db)


__all__ = ["Platform", "PlatformDB", "Audit", "Auth", "AuthError", "User", "Billing", "BillingError", "Keys", "KeyError_", "Conversations", "ConversationError"]
