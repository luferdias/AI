"""Configuração compartilhada dos testes: path, cliente e autenticação."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Permite os imports planos (from models import ...) usados na pasta api/.
API_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "api")
sys.path.insert(0, API_DIR)

# Senha de demonstração previsível para os testes.
os.environ.setdefault("USUARIO_DEMO_SENHA", "fiscal123")

from main import app  # noqa: E402  (após ajustar o sys.path)


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_header(client: TestClient) -> dict:
    resp = client.post("/token", data={"username": "fiscal", "password": "fiscal123"})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
