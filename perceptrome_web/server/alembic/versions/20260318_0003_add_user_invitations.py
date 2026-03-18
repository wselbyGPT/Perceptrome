"""add user invitations

Revision ID: 20260318_0003
Revises: 20260318_0002
Create Date: 2026-03-18 00:00:03
"""

from alembic import op
import sqlalchemy as sa


revision = "20260318_0003"
down_revision = "20260318_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_invitations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False, server_default=sa.text("'user'")),
        sa.Column("invited_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["invited_by_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(op.f("ix_user_invitations_created_at"), "user_invitations", ["created_at"], unique=False)
    op.create_index(op.f("ix_user_invitations_email"), "user_invitations", ["email"], unique=False)
    op.create_index(op.f("ix_user_invitations_expires_at"), "user_invitations", ["expires_at"], unique=False)
    op.create_index(op.f("ix_user_invitations_invited_by_user_id"), "user_invitations", ["invited_by_user_id"], unique=False)
    op.create_index(op.f("ix_user_invitations_token_hash"), "user_invitations", ["token_hash"], unique=False)
    op.create_index("ix_user_invitations_email_created", "user_invitations", ["email", "created_at"], unique=False)
    op.create_index("ix_user_invitations_token_lookup", "user_invitations", ["token_hash", "expires_at", "revoked_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_invitations_token_lookup", table_name="user_invitations")
    op.drop_index("ix_user_invitations_email_created", table_name="user_invitations")
    op.drop_index(op.f("ix_user_invitations_token_hash"), table_name="user_invitations")
    op.drop_index(op.f("ix_user_invitations_invited_by_user_id"), table_name="user_invitations")
    op.drop_index(op.f("ix_user_invitations_expires_at"), table_name="user_invitations")
    op.drop_index(op.f("ix_user_invitations_email"), table_name="user_invitations")
    op.drop_index(op.f("ix_user_invitations_created_at"), table_name="user_invitations")
    op.drop_table("user_invitations")
