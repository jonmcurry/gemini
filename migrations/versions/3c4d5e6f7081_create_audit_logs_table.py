"""Create audit_logs table

Revision ID: 3c4d5e6f7081
Revises: 2b3c4d5e6f70
Create Date: 2024-07-16 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql # For JSONB

# revision identifiers, used by Alembic.
revision = '3c4d5e6f7081'
down_revision = '2b3c4d5e6f70' # Points to the partitioning migration
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('timestamp', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=True),
        sa.Column('action', sa.String(length=255), nullable=False),
        sa.Column('resource', sa.String(length=255), nullable=True),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('patient_id_hash', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=100), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_fallback=True), nullable=True), # Use JSONB for PostgreSQL

        sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_logs'))
    )
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False) # Index on PK is often automatic, but explicit if model had index=True
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource'), 'audit_logs', ['resource'], unique=False)
    op.create_index(op.f('ix_audit_logs_resource_id'), 'audit_logs', ['resource_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_patient_id_hash'), 'audit_logs', ['patient_id_hash'], unique=False)
    op.create_index(op.f('ix_audit_logs_session_id'), 'audit_logs', ['session_id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_audit_logs_session_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_patient_id_hash'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_resource_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_resource'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_action'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_id'), table_name='audit_logs')
    op.drop_table('audit_logs')
