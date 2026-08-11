import uuid

from fastapi import Depends, FastAPI, HTTPException, status, APIRouter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Jobs
from app.schemas import JobCreate, JobResponse

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],    
)

@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job(job: JobCreate, db: Session = Depends(get_db)):
    new_job = Jobs(
        document_id=job.document_id,
        status=job.status
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    return new_job

@router.get("", response_model=list[JobResponse])
def get_jobs(db: Session = Depends(get_db)):
    statement = select(Jobs).order_by(Jobs.created_at.desc())
    jobs = db.scalars(statement).all()
    return jobs

@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    job = db.get(Jobs, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job

@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    job = db.get(Jobs, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    db.delete(job)
    db.commit()
    return None