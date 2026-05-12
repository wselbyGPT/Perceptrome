"""add model registry

Revision ID: 20260318_0004
Revises: 20260318_0003
Create Date: 2026-03-18 00:00:04
"""

from alembic import op
import sqlalchemy as sa


revision = "20260318_0004"
down_revision = "20260318_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "registered_models",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("visibility", sa.String(length=32), nullable=False, server_default=sa.text("'private'")),
        sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'active'")),
        sa.Column("tags_json", sa.Text(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("current_version_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_registered_models_created_at"), "registered_models", ["created_at"], unique=False)
    op.create_index(op.f("ix_registered_models_name"), "registered_models", ["name"], unique=False)
    op.create_index(op.f("ix_registered_models_owner_user_id"), "registered_models", ["owner_user_id"], unique=False)
    op.create_index(op.f("ix_registered_models_status"), "registered_models", ["status"], unique=False)
    op.create_index(op.f("ix_registered_models_updated_at"), "registered_models", ["updated_at"], unique=False)
    op.create_index(op.f("ix_registered_models_visibility"), "registered_models", ["visibility"], unique=False)
    op.create_index("ix_registered_models_owner_updated", "registered_models", ["owner_user_id", "updated_at"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_id", sa.String(length=36), nullable=False),
        sa.Column("source_run_id", sa.String(length=128), nullable=True),
        sa.Column("version_label", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'candidate'")),
        sa.Column("architecture", sa.String(length=80), nullable=True),
        sa.Column("tokenizer", sa.String(length=32), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=1024), nullable=True),
        sa.Column("config_snapshot_path", sa.String(length=1024), nullable=True),
        sa.Column("manifest_path", sa.String(length=1024), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("promoted_at", sa.DateTime(timezone=False), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["registered_models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_versions_architecture"), "model_versions", ["architecture"], unique=False)
    op.create_index(op.f("ix_model_versions_created_at"), "model_versions", ["created_at"], unique=False)
    op.create_index(op.f("ix_model_versions_model_id"), "model_versions", ["model_id"], unique=False)
    op.create_index(op.f("ix_model_versions_source_run_id"), "model_versions", ["source_run_id"], unique=False)
    op.create_index(op.f("ix_model_versions_status"), "model_versions", ["status"], unique=False)
    op.create_index(op.f("ix_model_versions_tokenizer"), "model_versions", ["tokenizer"], unique=False)
    op.create_index("ix_model_versions_model_created", "model_versions", ["model_id", "created_at"], unique=False)

    op.create_table(
        "model_version_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=80), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["version_id"], ["model_versions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_model_version_artifacts_created_at"), "model_version_artifacts", ["created_at"], unique=False)
    op.create_index(op.f("ix_model_version_artifacts_role"), "model_version_artifacts", ["role"], unique=False)
    op.create_index(op.f("ix_model_version_artifacts_version_id"), "model_version_artifacts", ["version_id"], unique=False)
    op.create_index("ix_model_version_artifacts_version_created", "model_version_artifacts", ["version_id", "created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_model_version_artifacts_version_created", table_name="model_version_artifacts")
    op.drop_index(op.f("ix_model_version_artifacts_version_id"), table_name="model_version_artifacts")
    op.drop_index(op.f("ix_model_version_artifacts_role"), table_name="model_version_artifacts")
    op.drop_index(op.f("ix_model_version_artifacts_created_at"), table_name="model_version_artifacts")
    op.drop_table("model_version_artifacts")
    op.drop_index("ix_model_versions_model_created", table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_tokenizer"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_status"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_source_run_id"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_model_id"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_created_at"), table_name="model_versions")
    op.drop_index(op.f("ix_model_versions_architecture"), table_name="model_versions")
    op.drop_table("model_versions")
    op.drop_index("ix_registered_models_owner_updated", table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_visibility"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_updated_at"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_status"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_owner_user_id"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_name"), table_name="registered_models")
    op.drop_index(op.f("ix_registered_models_created_at"), table_name="registered_models")
    op.drop_table("registered_models")
