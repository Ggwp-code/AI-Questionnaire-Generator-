"""Record when an outline topic drops out of its document's rebuilt outline."""
import sqlalchemy as sa
from alembic import op

import tribunal.db.types

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("topics", sa.Column("retired_at", tribunal.db.types.UTCDateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("topics", "retired_at")
