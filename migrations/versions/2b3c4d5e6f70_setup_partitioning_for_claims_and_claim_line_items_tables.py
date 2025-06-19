"""Setup partitioning for claims and claim_line_items tables

Revision ID: 2b3c4d5e6f70
Revises: 1a2b3c4d5e6f
Create Date: 2024-07-16 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2b3c4d5e6f70'
down_revision = '1a2b3c4d5e6f' # Points to the previous migration
branch_labels = None
depends_on = None

def upgrade():
    # Drop child table first, then parent.
    op.drop_table('claim_line_items')
    op.drop_table('claims')

    # Re-create 'claims' table with partitioning and composite PK
    op.create_table('claims',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_id', sa.String(length=100), nullable=False),
        sa.Column('facility_id', sa.String(length=50), nullable=False),
        sa.Column('patient_account_number', sa.String(length=100), nullable=False),
        sa.Column('patient_first_name', sa.String(length=100), nullable=True),
        sa.Column('patient_last_name', sa.String(length=100), nullable=True),
        sa.Column('patient_date_of_birth', sa.Date(), nullable=True),
        sa.Column('service_from_date', sa.Date(), nullable=False), # Partition Key
        sa.Column('service_to_date', sa.Date(), nullable=False),
        sa.Column('total_charges', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('processing_status', sa.String(length=50), server_default='pending', nullable=True),
        sa.Column('batch_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False), # onupdate handled by model
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id', 'service_from_date', name='pk_claims'),
        sa.UniqueConstraint('claim_id', name='uq_claims_claim_id'),
        postgresql_partition_by='RANGE (service_from_date)' # Direct kwarg for op.create_table
    )
    op.create_index(op.f('ix_claims_batch_id'), 'claims', ['batch_id'], unique=False)
    op.create_index(op.f('ix_claims_facility_id'), 'claims', ['facility_id'], unique=False)
    op.create_index(op.f('ix_claims_patient_account_number'), 'claims', ['patient_account_number'], unique=False)
    op.create_index(op.f('ix_claims_processing_status'), 'claims', ['processing_status'], unique=False)
    op.create_index(op.f('ix_claims_service_from_date'), 'claims', ['service_from_date'], unique=False) # Index on partition key
    op.create_index(op.f('ix_claims_service_to_date'), 'claims', ['service_to_date'], unique=False)
    # Note: Index on 'id' and 'claim_id' are implicitly created by PK and UQ constraints.

    # Re-create 'claim_line_items' table with partitioning and composite PK
    op.create_table('claim_line_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('claim_db_id', sa.Integer(), nullable=False), # This will be the FK to claims.id
        sa.Column('line_number', sa.Integer(), nullable=False),
        sa.Column('service_date', sa.Date(), nullable=False), # Partition Key
        sa.Column('procedure_code', sa.String(length=20), nullable=False),
        sa.Column('units', sa.Integer(), server_default='1', nullable=False),
        sa.Column('charge_amount', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('rvu_total', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False), # onupdate handled by model
        # Assuming 'claims.id' remains globally unique (e.g., from a sequence).
        # The FK can still point to just 'claims.id'. PostgreSQL allows FKs to unique constrained columns,
        # and a PK column is implicitly unique. For composite PK on parent, if FK is only to one part,
        # that part must have a unique constraint itself. Here, 'claims.id' is not unique alone in the new PK.
        # However, if 'id' was previously SERIAL PK, it likely has a sequence making its values unique.
        # A more robust FK to a partitioned table often includes the partition key of the parent.
        # For now, sticking to simple FK to claims.id as per typical non-partitioned setup.
        # This might need adjustment if DB enforces FKs to span all parent PK columns for partitioned tables.
        # Let's assume for this step that claims.id is unique and can be referenced.
        # The previous migration had: ForeignKeyConstraint(['claim_db_id'], ['claims.id'], name='fk_line_items_to_claims_id')
        # Recreating a similar one for now.
        sa.ForeignKeyConstraint(['claim_db_id'], ['claims.id'], name='fk_claim_line_items_claim_db_id_claims_id'),
        sa.PrimaryKeyConstraint('id', 'service_date', name='pk_claim_line_items'),
        postgresql_partition_by='RANGE (service_date)' # Direct kwarg
    )
    op.create_index(op.f('ix_claim_line_items_claim_db_id'), 'claim_line_items', ['claim_db_id'], unique=False)
    op.create_index(op.f('ix_claim_line_items_procedure_code'), 'claim_line_items', ['procedure_code'], unique=False)
    op.create_index(op.f('ix_claim_line_items_service_date'), 'claim_line_items', ['service_date'], unique=False) # Index on partition key

    # Create initial partitions
    op.execute("CREATE TABLE claims_y2023 PARTITION OF claims FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')")
    op.execute("CREATE TABLE claims_y2024 PARTITION OF claims FOR VALUES FROM ('2024-01-01') TO ('2025-01-01')")
    op.execute("CREATE TABLE claims_y2025 PARTITION OF claims FOR VALUES FROM ('2025-01-01') TO ('2026-01-01')")

    op.execute("CREATE TABLE claim_line_items_y2024m01 PARTITION OF claim_line_items FOR VALUES FROM ('2024-01-01') TO ('2024-02-01')")
    op.execute("CREATE TABLE claim_line_items_y2024m02 PARTITION OF claim_line_items FOR VALUES FROM ('2024-02-01') TO ('2024-03-01')")
    op.execute("CREATE TABLE claim_line_items_y2024m03 PARTITION OF claim_line_items FOR VALUES FROM ('2024-03-01') TO ('2024-04-01')")

def downgrade():
    # Drop partitions first
    op.execute("DROP TABLE IF EXISTS claim_line_items_y2024m01")
    op.execute("DROP TABLE IF EXISTS claim_line_items_y2024m02")
    op.execute("DROP TABLE IF EXISTS claim_line_items_y2024m03")
    op.execute("DROP TABLE IF EXISTS claims_y2023")
    op.execute("DROP TABLE IF EXISTS claims_y2024")
    op.execute("DROP TABLE IF EXISTS claims_y2025")

    # Drop the partitioned (parent) tables
    op.drop_table('claim_line_items')
    op.drop_table('claims')

    # Recreate tables as they were in migration '1a2b3c4d5e6f'
    op.create_table('claims',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('claim_id', sa.String(length=100), nullable=False), # unique=True implied by create_index below
        sa.Column('facility_id', sa.String(length=50), nullable=False),
        sa.Column('patient_account_number', sa.String(length=100), nullable=False),
        sa.Column('patient_first_name', sa.String(length=100), nullable=True),
        sa.Column('patient_last_name', sa.String(length=100), nullable=True),
        sa.Column('patient_date_of_birth', sa.Date(), nullable=True),
        sa.Column('service_from_date', sa.Date(), nullable=False),
        sa.Column('service_to_date', sa.Date(), nullable=False),
        sa.Column('total_charges', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('processing_status', sa.String(length=50), server_default='pending', nullable=True),
        sa.Column('batch_id', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processed_at', sa.TIMESTAMP(timezone=True), nullable=True)
    )
    # Original indexes (from 1a2b3c4d5e6f script)
    op.create_index(op.f('ix_claims_batch_id'), 'claims', ['batch_id'], unique=False)
    op.create_index(op.f('ix_claims_claim_id'), 'claims', ['claim_id'], unique=True)
    op.create_index(op.f('ix_claims_facility_id'), 'claims', ['facility_id'], unique=False)
    op.create_index(op.f('ix_claims_id'), 'claims', ['id'], unique=False) # This was in original
    op.create_index(op.f('ix_claims_patient_account_number'), 'claims', ['patient_account_number'], unique=False)
    op.create_index(op.f('ix_claims_processing_status'), 'claims', ['processing_status'], unique=False)
    op.create_index(op.f('ix_claims_service_from_date'), 'claims', ['service_from_date'], unique=False)
    op.create_index(op.f('ix_claims_service_to_date'), 'claims', ['service_to_date'], unique=False)

    op.create_table('claim_line_items',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('claim_db_id', sa.Integer(), nullable=False),
        sa.Column('line_number', sa.Integer(), nullable=False),
        sa.Column('service_date', sa.Date(), nullable=False),
        sa.Column('procedure_code', sa.String(length=20), nullable=False),
        sa.Column('units', sa.Integer(), server_default='1', nullable=False),
        sa.Column('charge_amount', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('rvu_total', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['claim_db_id'], ['claims.id'], name='fk_claim_line_items_claim_db_id_claims_id') # Recreate original FK
    )
    # Original indexes (from 1a2b3c4d5e6f script)
    op.create_index(op.f('ix_claim_line_items_claim_db_id'), 'claim_line_items', ['claim_db_id'], unique=False)
    op.create_index(op.f('ix_claim_line_items_id'), 'claim_line_items', ['id'], unique=False) # This was in original
    op.create_index(op.f('ix_claim_line_items_procedure_code'), 'claim_line_items', ['procedure_code'], unique=False)
    op.create_index(op.f('ix_claim_line_items_service_date'), 'claim_line_items', ['service_date'], unique=False)
