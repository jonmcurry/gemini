"""Create initial claims and claim_line_items tables

Revision ID: 1a2b3c4d5e6f
Revises:
Create Date: 2024-07-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql # For specific PG types if needed

# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('claims',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_id', sa.String(length=100), nullable=False),
        sa.Column('facility_id', sa.String(length=50), nullable=False),
        sa.Column('patient_account_number', sa.String(length=100), nullable=False),
        sa.Column('patient_first_name', sa.String(length=100), nullable=True),
        sa.Column('patient_last_name', sa.String(length=100), nullable=True),
        sa.Column('patient_date_of_birth', sa.Date(), nullable=True),
        sa.Column('service_from_date', sa.Date(), nullable=False),
        sa.Column('service_to_date', sa.Date(), nullable=False),
        sa.Column('total_charges', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('processing_status', sa.String(length=50), server_default='pending', nullable=True), # Alembic often makes server_default nullable=True
        sa.Column('batch_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False), # onupdate handled by SQLAlchemy model's onupdate=func.now()
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_claims_batch_id'), 'claims', ['batch_id'], unique=False)
    op.create_index(op.f('ix_claims_claim_id'), 'claims', ['claim_id'], unique=True)
    op.create_index(op.f('ix_claims_facility_id'), 'claims', ['facility_id'], unique=False)
    op.create_index(op.f('ix_claims_id'), 'claims', ['id'], unique=False)
    op.create_index(op.f('ix_claims_patient_account_number'), 'claims', ['patient_account_number'], unique=False)
    op.create_index(op.f('ix_claims_processing_status'), 'claims', ['processing_status'], unique=False)
    op.create_index(op.f('ix_claims_service_from_date'), 'claims', ['service_from_date'], unique=False)
    op.create_index(op.f('ix_claims_service_to_date'), 'claims', ['service_to_date'], unique=False)

    op.create_table('claim_line_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_db_id', sa.Integer(), nullable=False),
        sa.Column('line_number', sa.Integer(), nullable=False),
        sa.Column('service_date', sa.Date(), nullable=False),
        sa.Column('procedure_code', sa.String(length=20), nullable=False),
        sa.Column('units', sa.Integer(), server_default='1', nullable=False), # Model has default=1, nullable=False. server_default='1' is appropriate.
        sa.Column('charge_amount', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('rvu_total', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False), # onupdate handled by SQLAlchemy model
        sa.ForeignKeyConstraint(['claim_db_id'], ['claims.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_claim_line_items_claim_db_id'), 'claim_line_items', ['claim_db_id'], unique=False)
    op.create_index(op.f('ix_claim_line_items_id'), 'claim_line_items', ['id'], unique=False)
    op.create_index(op.f('ix_claim_line_items_procedure_code'), 'claim_line_items', ['procedure_code'], unique=False)
    op.create_index(op.f('ix_claim_line_items_service_date'), 'claim_line_items', ['service_date'], unique=False)
    # ### end Alembic commands ###

def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_claim_line_items_service_date'), table_name='claim_line_items')
    op.drop_index(op.f('ix_claim_line_items_procedure_code'), table_name='claim_line_items')
    op.drop_index(op.f('ix_claim_line_items_id'), table_name='claim_line_items')
    op.drop_index(op.f('ix_claim_line_items_claim_db_id'), table_name='claim_line_items')
    op.drop_table('claim_line_items')

    op.drop_index(op.f('ix_claims_service_to_date'), table_name='claims')
    op.drop_index(op.f('ix_claims_service_from_date'), table_name='claims')
    op.drop_index(op.f('ix_claims_processing_status'), table_name='claims')
    op.drop_index(op.f('ix_claims_patient_account_number'), table_name='claims')
    # op.drop_index(op.f('ix_claims_id'), table_name='claims') # This index is on PK, usually not explicitly dropped unless create_index was used. Autogen might or might not drop it.
    op.drop_index(op.f('ix_claims_facility_id'), table_name='claims')
    op.drop_index(op.f('ix_claims_claim_id'), table_name='claims')
    op.drop_index(op.f('ix_claims_batch_id'), table_name='claims')
    op.drop_table('claims')
    # ### end Alembic commands ###
