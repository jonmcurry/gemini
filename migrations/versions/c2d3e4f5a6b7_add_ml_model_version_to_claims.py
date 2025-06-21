"""Add ml_model_version_used to claims table

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2024-03-15 10:05:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c2d3e4f5a6b7'
down_revision = 'b1c2d3e4f5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('claims',
                  sa.Column('ml_model_version_used', sa.String(length=50), nullable=True)
                 )
    op.create_index(op.f('ix_claims_ml_model_version_used'), 'claims', ['ml_model_version_used'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_claims_ml_model_version_used'), table_name='claims')
    op.drop_column('claims', 'ml_model_version_used')
