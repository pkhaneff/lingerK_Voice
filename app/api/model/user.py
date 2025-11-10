import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.api.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = sa.Column(UUID(as_uuid=True), primary_key=True,
                   server_default=sa.text("gen_random_uuid()"))
    email = sa.Column(sa.String(255), nullable=False)
    hashed_password = sa.Column(sa.String(255), nullable=False)
    first_name = sa.Column(sa.String(100), nullable=True)
    last_name = sa.Column(sa.String(100), nullable=True)
    is_superuser = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false"))
    is_active = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("true"))
    is_verified = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false"))
    verified_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    last_login_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    failed_login_attempts = sa.Column(sa.Integer, nullable=False, server_default=sa.text("0"))
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    updated_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))

    api_keys = relationship("ApiKey", back_populates="user")

    __table_args__ = (
        sa.Index("ix_users_email", "email", unique=True),
    )