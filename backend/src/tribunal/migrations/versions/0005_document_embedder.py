"""Record which embedder indexed each document."""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("embedder_id", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "embedder_id")
