# This file ensures that all models are imported when Base.metadata is accessed.
# Alembic's env.py can then simply import this module or Base directly if models are loaded.

# Assuming Base is defined in db_session.py, which is one level up from 'models' directory.
# If Base is defined elsewhere, adjust the import.
# from ..db_session import Base # Not strictly necessary to re-export Base if env.py gets it from source

from .claims_db import ClaimModel, ClaimLineItemModel
from .audit_log_db import AuditLogModel
from .claims_production_db import ClaimsProductionModel # Added this line

# Optional: Define __all__ to control what `from .models import *` imports
# __all__ = [
#     "ClaimModel",
#     "ClaimLineItemModel",
#     "AuditLogModel",
#     "ClaimsProductionModel", # Added this line
#     # "Base" # if re-exporting Base
# ]
