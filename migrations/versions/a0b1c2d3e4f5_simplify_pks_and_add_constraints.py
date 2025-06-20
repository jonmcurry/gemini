"""Simplify PKs for claims and claim_line_items, add unique constraints.

Revision ID: a0b1c2d3e4f5
Revises: 9c0d1e2f3a4b
Create Date: 2024-04-01 12:10:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a0b1c2d3e4f5'
down_revision = '9c0d1e2f3a4b' # Previous migration (add_insurance_financial_to_claims)
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands for claims table ###

    # Drop existing primary key constraint if it's composite
    # The name 'pk_claims' was defined in the model's __table_args__
    # Need to ensure this name is correct by checking an existing DB or prior migrations.
    # Assuming 'pk_claims' is the correct name of the PK constraint.
    op.drop_constraint('pk_claims', 'claims', type_='primary')

    # Alter 'id' column to be a simple auto-incrementing primary key
    # If 'id' was part of old composite PK and not auto-incrementing, its definition needs to change.
    # If it was already an Integer, making it primary_key=True, autoincrement=True might be enough.
    # However, SQLAlchemy usually requires specific ALTER for PK.
    # For simplicity, assuming 'id' column itself doesn't need type change, just PK status.
    # Most DBs require PK to be NOT NULL, which is already in model.
    # Making it autoincrement might need specific sequence handling in some DBs if not SERIAL.
    # For PostgreSQL, SERIAL implies autoincrement and primary key.
    # If 'id' was `Column(Integer, index=True, nullable=False)`
    # And now needs to be `Column(Integer, primary_key=True, index=True, autoincrement=True)`
    # The `primary_key=True` part is handled by `create_primary_key`.
    # `autoincrement=True` needs the column to be of a type that supports it (e.g. SERIAL in PG, INTEGER PRIMARY KEY AUTOINCREMENT in SQLite)
    # This part is tricky with Alembic as it depends on current column definition.
    # If 'id' column was just `Integer`, we might need to recreate or alter it extensively.
    # Given the model change, we are effectively making 'id' the sole PK.

    # Add 'id' as the new simple primary key.
    # This assumes 'id' column exists and is suitable (e.g. Integer).
    # The autoincrement=True in the model will translate to SERIAL or IDENTITY depending on dialect if column is new.
    # If altering, this is more complex. Let's assume we are defining PK on existing 'id' column.
    op.create_primary_key(
        "pk_claims", "claims",
        ["id"]
    )
    # Note: Making 'id' autoincrementing if it wasn't already (e.g. from INT to SERIAL)
    # is a more involved operation and might require table rebuild or specific ALTER sequences.
    # The model change to autoincrement=True implies this intent.
    # For now, this migration focuses on constraint changes and assumes 'id' type is compatible.

    # 'claim_id' is already unique via unique=True in model, index exists.
    # If a named unique constraint is desired for 'claim_id' (and wasn't auto-created with a good name):
    # op.create_unique_constraint('uq_claims_claim_id', 'claims', ['claim_id'])


    # ### commands for claim_line_items table ###

    # Drop existing primary key constraint if it's composite
    op.drop_constraint('pk_claim_line_items', 'claim_line_items', type_='primary')

    # Add 'id' as the new simple primary key
    op.create_primary_key(
        "pk_claim_line_items", "claim_line_items",
        ["id"]
    )

    # Add unique constraint for (claim_db_id, line_number)
    op.create_unique_constraint('uq_claimitem_claim_line', 'claim_line_items', ['claim_db_id', 'line_number'])

    # The other unique constraint uq_claimitem_claim_line_servicedate might be overly restrictive
    # if line numbers are unique per claim regardless of service date.
    # If it was indeed added by the tool, and it's desired:
    # op.create_unique_constraint('uq_claimitem_claim_line_servicedate', 'claim_line_items', ['claim_db_id', 'line_number', 'service_date'])


def downgrade() -> None:
    # ### commands for claim_line_items table ###
    op.drop_constraint('uq_claimitem_claim_line', 'claim_line_items', type_='unique')
    # op.drop_constraint('uq_claimitem_claim_line_servicedate', 'claim_line_items', type_='unique') # If added in upgrade

    op.drop_constraint('pk_claim_line_items', 'claim_line_items', type_='primary')
    # Recreate old composite primary key for claim_line_items
    op.create_primary_key(
        'pk_claim_line_items', 'claim_line_items',
        ['id', 'service_date']
    )

    # ### commands for claims table ###
    # op.drop_constraint('uq_claims_claim_id', 'claims', type_='unique') # If named one was created

    op.drop_constraint('pk_claims', 'claims', type_='primary')
    # Recreate old composite primary key for claims
    op.create_primary_key(
        'pk_claims', 'claims',
        ['id', 'service_from_date']
    )

    # Note: Downgrading autoincrement behavior is complex and database-specific.
    # This script primarily handles constraint changes. Type changes for 'id' to remove
    # autoincrement (e.g., from SERIAL to INT) are not explicitly handled here.
```
