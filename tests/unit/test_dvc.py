"""Unit tests for DVC integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ingestion_agent.dvc_integration import DVCManager, run_dvc_add, run_dvc_push


@pytest.fixture
def dvc_manager():
    """Create DVC manager instance."""
    return DVCManager(remote_name="test-remote")


@pytest.mark.asyncio
async def test_dvc_add_success(dvc_manager):
    """Test successful DVC add operation."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        # Mock dvc add
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        # Create temp file
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        Path(test_file).touch()
        Path(f"{test_file}.dvc").touch()

        result = await dvc_manager.add(test_file)

        assert result is True
        assert mock_exec.call_count >= 1


@pytest.mark.asyncio
async def test_dvc_add_failure(dvc_manager):
    """Test failed DVC add operation."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        result = await dvc_manager.add("/nonexistent/file")

        assert result is False


@pytest.mark.asyncio
async def test_dvc_push_success(dvc_manager):
    """Test successful DVC push operation."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        result = await dvc_manager.push()

        assert result is True


@pytest.mark.asyncio
async def test_dvc_push_with_remote(dvc_manager):
    """Test DVC push with specific remote."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        result = await dvc_manager.push(remote="custom-remote")

        assert result is True


@pytest.mark.asyncio
async def test_dvc_push_failure(dvc_manager):
    """Test failed DVC push operation."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"push error")
        mock_process.returncode = 1
        mock_exec.return_value = mock_process

        result = await dvc_manager.push()

        assert result is False


@pytest.mark.asyncio
async def test_init_remote(dvc_manager):
    """Test DVC remote initialization."""
    with patch("asyncio.create_subprocess_exec") as mock_exec:
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_exec.return_value = mock_process

        result = await dvc_manager.init_remote("s3://my-bucket/dvc-storage")

        assert result is True


def test_run_dvc_add_sync():
    """Test synchronous DVC add wrapper."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.txt")
        Path(test_file).touch()

        result = run_dvc_add(test_file)

        assert result is True
        assert mock_run.call_count == 2  # dvc add + git add


def test_run_dvc_push_sync():
    """Test synchronous DVC push wrapper."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0)

        result = run_dvc_push()

        assert result is True
