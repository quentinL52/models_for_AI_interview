from contextvars import ContextVar
from typing import Dict, Any

_cv_document_context = ContextVar('cv_document_context', default=None)
_job_offer_context = ContextVar('job_offer_context', default=None)

def set_current_interview_context(cv_document: Dict[str, Any], job_offer: Dict[str, Any]):
    _cv_document_context.set(cv_document)
    _job_offer_context.set(job_offer)

def get_current_interview_context():
    return _cv_document_context.get(), _job_offer_context.get()