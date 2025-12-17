import pytest
from core.health import health_checker, ServiceStatus

@pytest.mark.asyncio
async def test_pinecone():
    """Test Pinecone vector store health check"""

    await health_checker.add_service(
        "pinecone",
        health_checker.check_pinecone,
        timeout=10.0
    )

    result = await health_checker.check_service_health("pinecone")
    assert result == ServiceStatus.HEALTHY, f"Pinecone health check failed: {result}"
    print(f"Pinecone health: {result}")

