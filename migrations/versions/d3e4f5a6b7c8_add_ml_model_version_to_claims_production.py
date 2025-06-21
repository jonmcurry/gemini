"""Add ml_model_version_used to claims_production table

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2024-03-15 10:10:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd3e4f5a6b7c8'
down_revision = 'c2d3e4f5a6b7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('claims_production',
                  sa.Column('ml_model_version_used', sa.String(length=50), nullable=True)
                 )
    op.create_index(op.f('ix_claims_production_ml_model_version_used'), 'claims_production', ['ml_model_version_used'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_claims_production_ml_model_version_used'), table_name='claims_production')
    op.drop_column('claims_production', 'ml_model_version_used')
