import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from backend.app.db.session import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    columns = Column(JSON, nullable=False)
    schema_details = Column(JSON, nullable=True)
    detected_mode = Column(String(50), nullable=True, default="pending")
    mode_reason = Column(Text, nullable=True)
    match_details = Column(JSON, nullable=True)
    status = Column(String(50), nullable=False, default="uploaded")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "schema_details": self.schema_details,
            "detected_mode": self.detected_mode,
            "mode_reason": self.mode_reason,
            "match_details": self.match_details,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }



class CleaningLog(Base):
    __tablename__ = "cleaning_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), nullable=False)
    action = Column(String(50), nullable=False)  # e.g., 'drop', 'impute', 'duplicate_remove', 'suspicious_flag'
    details = Column(JSON, nullable=True)  # free-form details like columns affected, rows count, reason
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "action": self.action,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
