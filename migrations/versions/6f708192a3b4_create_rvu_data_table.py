"""Create rvu_data table

Revision ID: 6f708192a3b4
Revises: 5e6f708192a3
Create Date: 2024-07-16 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
# No postgresql specific types needed for this table directly in this migration.
from decimal import Decimal # For seeding data

# revision identifiers, used by Alembic.
revision = '6f708192a3b4'
down_revision = '5e6f708192a3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('rvu_data',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('procedure_code', sa.String(length=50), nullable=False),
        sa.Column('rvu_value', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        # Constraints and indexes defined below for clarity and using op.f()
    )
    # Primary key is already defined on the id column.
    # Index on id is typically created automatically for PK, but explicit op.f() ensures consistent naming if needed.
    op.create_index(op.f('ix_rvu_data_id'), 'rvu_data', ['id'], unique=False)
    # Unique constraint and index for procedure_code
    op.create_unique_constraint(op.f('uq_rvu_data_procedure_code'), 'rvu_data', ['procedure_code'])
    op.create_index(op.f('ix_rvu_data_procedure_code'), 'rvu_data', ['procedure_code'], unique=True)

    # Seed initial data
    rvu_table = sa.Table('rvu_data', sa.MetaData(),
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True),
        sa.Column('procedure_code', sa.String(length=50), nullable=False),
        sa.Column('rvu_value', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('description', sa.Text(), nullable=True)
    )

    rvu_seed_data = [
        {'procedure_code': '99213', 'rvu_value': Decimal('2.11'), 'description': 'Office visit, established patient, level 3'},
        {'procedure_code': '99214', 'rvu_value': Decimal('3.28'), 'description': 'Office visit, established patient, level 4'},
        {'procedure_code': '80053', 'rvu_value': Decimal('0.85'), 'description': 'Comprehensive metabolic panel'},
        {'procedure_code': '99203', 'rvu_value': Decimal('1.92'), 'description': 'Office visit, new patient, level 3'},
        {'procedure_code': 'DEFAULT_RVU', 'rvu_value': Decimal('1.00'), 'description': 'Default RVU for unlisted or unspecified procedures'}
    ]
    op.bulk_insert(rvu_table, rvu_seed_data)


def downgrade():
    # Delete seeded data first (optional, as table drop will remove it, but good for precision)
    # Matching the specific procedure codes ensures only these seeded entries are targeted if other data existed.
    op.execute("DELETE FROM rvu_data WHERE procedure_code IN ('99213', '99214', '80053', '99203', 'DEFAULT_RVU')")

    op.drop_index(op.f('ix_rvu_data_procedure_code'), table_name='rvu_data')
    op.drop_constraint(op.f('uq_rvu_data_procedure_code'), 'rvu_data', type_='unique')
    op.drop_index(op.f('ix_rvu_data_id'), table_name='rvu_data')
    op.drop_table('rvu_data')
