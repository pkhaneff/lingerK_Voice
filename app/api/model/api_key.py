from sqlalchemy.dialects.postgresql import UUID

from sqlalchemy.orm import relationship
from app.api.db.base import Base
import sqlalchemy as sa

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = sa.Column(UUID(as_uuid=True), primary_key=True,
                   server_default=sa.text("gen_random_uuid()"))
    user_id = sa.Column(UUID(as_uuid=True),sa.ForeignKey("users.id"),nullable=False)
    hashed_key = sa.Column(sa.String(255), nullable=False)
    prefix = sa.Column(sa.String(8), nullable=False)
    key_hint = sa.Column(sa.String(4), nullable=False)
    description = sa.Column(sa.String(255), nullable=True)
    is_active = sa.Column(sa.Boolean, nullable=False, server_default=sa.text("true"))
    last_used_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), nullable=False,
                          server_default=sa.text("NOW()"))
    updated_at = sa.Column(sa.DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")