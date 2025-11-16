"""DVC integration module for data versioning."""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

from prometheus_client import Counter

logger = logging.getLogger("ingestion_agent.dvc")

# Metrics
dvc_add_success = Counter("dvc_add_success_total", "Successful DVC add operations")
dvc_add_failed = Counter("dvc_add_failed_total", "Failed DVC add operations")
dvc_push_success = Counter("dvc_push_success_total", "Successful DVC push operations")
dvc_push_failed = Counter("dvc_push_failed_total", "Failed DVC push operations")


class DVCManager:
    """Manages DVC operations for data versioning."""
    
    def __init__(self, remote_name: str = "storage"):
        """Initialize DVC manager.
        
        Args:
            remote_name: Name of the DVC remote
        """
        self.remote_name = remote_name
        
    async def add(self, path: str) -> bool:
        """Add file or directory to DVC tracking.
        
        Args:
            path: Path to file or directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Run dvc add
            result = await asyncio.create_subprocess_exec(
                "dvc", "add", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"DVC add failed: {stderr.decode()}")
                dvc_add_failed.inc()
                return False
            
            # Add .dvc file to git
            dvc_file = f"{path}.dvc"
            if Path(dvc_file).exists():
                git_result = await asyncio.create_subprocess_exec(
                    "git", "add", dvc_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await git_result.communicate()
            
            dvc_add_success.inc()
            logger.info(f"Successfully added {path} to DVC")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to DVC: {e}")
            dvc_add_failed.inc()
            return False
    
    async def push(self, remote: Optional[str] = None) -> bool:
        """Push DVC-tracked data to remote storage.
        
        Args:
            remote: Optional remote name (uses default if not specified)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["dvc", "push"]
            if remote:
                cmd.extend(["-r", remote])
            elif self.remote_name:
                cmd.extend(["-r", self.remote_name])
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"DVC push failed: {stderr.decode()}")
                dvc_push_failed.inc()
                return False
            
            dvc_push_success.inc()
            logger.info("Successfully pushed to DVC remote")
            return True
            
        except Exception as e:
            logger.error(f"Error pushing to DVC: {e}")
            dvc_push_failed.inc()
            return False
    
    async def init_remote(self, remote_url: str, remote_name: Optional[str] = None) -> bool:
        """Initialize DVC remote configuration.
        
        Args:
            remote_url: URL of the remote storage
            remote_name: Name for the remote (uses default if not specified)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            name = remote_name or self.remote_name
            
            # Add remote
            result = await asyncio.create_subprocess_exec(
                "dvc", "remote", "add", "-d", name, remote_url,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                # Remote might already exist, try modify
                result = await asyncio.create_subprocess_exec(
                    "dvc", "remote", "modify", name, "url", remote_url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.communicate()
            
            logger.info(f"Initialized DVC remote {name} at {remote_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing DVC remote: {e}")
            return False


def run_dvc_add(path: str) -> bool:
    """Synchronous wrapper for DVC add operation.
    
    Args:
        path: Path to add to DVC
        
    Returns:
        True if successful, False otherwise
    """
    try:
        subprocess.run(["dvc", "add", path], check=True, capture_output=True)
        subprocess.run(["git", "add", f"{path}.dvc"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC add failed: {e.stderr.decode()}")
        return False


def run_dvc_push(remote: Optional[str] = None) -> bool:
    """Synchronous wrapper for DVC push operation.
    
    Args:
        remote: Optional remote name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = ["dvc", "push"]
        if remote:
            cmd.extend(["-r", remote])
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC push failed: {e.stderr.decode()}")
        return False
