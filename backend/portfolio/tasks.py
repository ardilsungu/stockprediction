import time
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def run_portfolio_optimization(self, job_id, params):
    from .models import PortfolioJob, PortfolioResult

    try:
        job = PortfolioJob.objects.get(id=job_id)
        job.status = 'running'
        job.celery_task_id = self.request.id
        job.save()

        logger.info(f"Portfolio optimization started: job_id={job_id}")

        # Mock optimizer — gerçek optimizer buraya bağlanacak
        time.sleep(2)

        assets = params.get('assets', ['BTC', 'ETH', 'SOL'])
        risk_tolerance = params.get('risk_tolerance', 'medium')

        mock_result = {
            'surviving_assets': assets,
            'pareto_solutions': [
                {'return': 0.15, 'risk': 0.10, 'sharpe': 1.50},
                {'return': 0.22, 'risk': 0.18, 'sharpe': 1.22},
                {'return': 0.30, 'risk': 0.28, 'sharpe': 1.07},
            ],
            'strategies': [
                {'name': 'Conservative', 'expected_return': 0.15, 'volatility': 0.10},
                {'name': 'Balanced',     'expected_return': 0.22, 'volatility': 0.18},
                {'name': 'Aggressive',   'expected_return': 0.30, 'volatility': 0.28},
            ],
            'weights': {
                asset: round(1 / len(assets), 4) for asset in assets
            },
        }

        PortfolioResult.objects.create(job=job, **mock_result)

        job.status = 'completed'
        job.save()

        logger.info(f"Portfolio optimization completed: job_id={job_id}")
        return {'status': 'completed', 'job_id': str(job_id)}

    except PortfolioJob.DoesNotExist:
        logger.error(f"Job not found: job_id={job_id}")
        raise

    except Exception as exc:
        logger.error(f"Portfolio optimization failed: job_id={job_id}, error={exc}")
        try:
            job = PortfolioJob.objects.get(id=job_id)
            job.status = 'failed'
            job.save()
        except Exception:
            pass
        raise self.retry(exc=exc)
