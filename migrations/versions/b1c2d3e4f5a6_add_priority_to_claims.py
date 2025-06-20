"""Add priority to claims table

Revision ID: b1c2d3e4f5a6
Revises: a0b1c2d3e4f5
Create Date: 2024-03-15 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'b1c2d3e4f5a6'
down_revision = 'a0b1c2d3e4f5' # Points to the previous head
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column('claims',
                  sa.Column('priority', sa.Integer(), server_default=sa.text('1'), nullable=False)
                 )
    op.create_index(op.f('ix_claims_priority'), 'claims', ['priority'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_claims_priority'), table_name='claims')
    op.drop_column('claims', 'priority')
