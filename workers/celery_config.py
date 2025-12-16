from celery import Celery
from core.config import settings

celery_app = Celery(
    "langgraph_research_analyzer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend_url,
    include=[
        "workers.tasks.paper_tasks",  # Paper analysis tasks
    ]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_time_limit=2100,
    task_soft_time_limit=1800,
    task_max_retries=3,
    task_default_retry_delay=300,
    task_routes={
        'workers.tasks.paper_tasks.*': {'queue': 'paper_analysis'},
    }
)