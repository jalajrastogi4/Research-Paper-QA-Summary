import pytest
from core.health import health_checker


@pytest.mark.asyncio
async def test_all_services():
    """Test health checks for all services including Pinecone"""

    services = [
        ("database", health_checker.check_database, 5.0),
        ("redis", health_checker.check_redis, 5.0),
        ("celery", health_checker.check_celery, 5.0),
        ("pinecone", health_checker.check_pinecone, 10.0),
    ]

    for name, check_func, timeout in services:
        await health_checker.add_service(name, check_func, timeout)

    status = await health_checker.check_all_services()

    for service_name, service_info in status['services'].items():
        assert service_info['status'] == 'healthy', (
            f"Service {service_name} is {service_info['status']} "
            f"with error: {service_info.get('error')}"
        )

    assert status["status"] == 'healthy', "Overall system health check failed"