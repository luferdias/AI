"""Testes do diagnóstico de demanda (regra) e da proteção por autenticação."""


def test_diagnostico_exige_autenticacao(client):
    resp = client.post("/api/v1/diagnostico/demanda", json={"dr": 400, "dc": 400})
    assert resp.status_code == 401


def test_diagnostico_ultrapassagem(client, auth_header):
    resp = client.post("/api/v1/diagnostico/demanda",
                       json={"dr": 429.84, "dc": 400.0}, headers=auth_header)
    assert resp.status_code == 200
    assert resp.json()["situacao"] == "demanda_ultrapassada"


def test_diagnostico_subutilizacao(client, auth_header):
    resp = client.post("/api/v1/diagnostico/demanda",
                       json={"dr": 360.72, "dc": 400.0}, headers=auth_header)
    assert resp.status_code == 200
    assert resp.json()["situacao"] == "demanda_nao_utilizada"


def test_diagnostico_eficiente_com_valor(client, auth_header):
    resp = client.post("/api/v1/diagnostico/demanda",
                       json={"dr": 400.0, "dc": 400.0, "tarifa_demanda": 39.82},
                       headers=auth_header)
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["situacao"] == "demanda_eficiente"
    assert corpo["valor_financeiro"] == 0.0


def test_diagnostico_entrada_invalida(client, auth_header):
    resp = client.post("/api/v1/diagnostico/demanda",
                       json={"dr": -10, "dc": 400}, headers=auth_header)
    assert resp.status_code == 422
