"""Add new fields to claims table for transfer and ml results

Revision ID: 5e6f708192a3
Revises: 4d5e6f708192
Create Date: 2024-07-16 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql # Not strictly needed here, but good habit

# revision identifiers, used by Alembic.
revision = '5e6f708192a3'
down_revision = '4d5e6f708192'
branch_labels = None
depends_on = None

def upgrade():
    # Add columns to 'claims' table
    op.add_column('claims', sa.Column('transferred_to_prod_at', sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column('claims', sa.Column('processing_duration_ms', sa.Integer(), nullable=True))
    op.add_column('claims', sa.Column('ml_score', sa.Numeric(precision=5, scale=4), nullable=True))
    op.add_column('claims', sa.Column('ml_derived_decision', sa.String(length=50), nullable=True))

    # Create indexes for new queryable/filterable columns if needed
    op.create_index(op.f('ix_claims_transferred_to_prod_at'), 'claims', ['transferred_to_prod_at'], unique=False)
    # Not indexing ml_score or ml_derived_decision for now unless they become frequent query targets.

def downgrade():
    # Drop indexes first
    op.drop_index(op.f('ix_claims_transferred_to_prod_at'), table_name='claims')

    # Drop columns from 'claims' table
    op.drop_column('claims', 'ml_derived_decision')
    op.drop_column('claims', 'ml_score')
    op.drop_column('claims', 'processing_duration_ms')
    op.drop_column('claims', 'transferred_to_prod_at')
