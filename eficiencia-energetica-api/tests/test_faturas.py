"""Testes dos endpoints de faturas e análise consolidada."""


def _fatura_valida(mes="2025-03"):
    return {
        "mes_ano": mes,
        "consumo_total_kwh": 77579.37,
        "demanda_contratada_kw": 400.0,
        "demanda_registrada_kw": 371.52,
        "bandeira": "verde",
        "geracao_fv_kwh": 10080.0,
        "fator_potencia": 0.993,
        "valor_fatura": 57493.57,
    }


def test_criar_fatura_valida(client):
    resp = client.post("/api/v1/faturas", json=_fatura_valida("2025-01"))
    assert resp.status_code == 201
    assert resp.json()["mes_ano"] == "2025-01"
    assert "id" in resp.json()


def test_criar_fatura_mes_invalido(client):
    dados = _fatura_valida()
    dados["mes_ano"] = "2025/13"  # formato e mês inválidos
    resp = client.post("/api/v1/faturas", json=dados)
    assert resp.status_code == 422


def test_criar_fatura_fator_potencia_invalido(client):
    dados = _fatura_valida("2025-02")
    dados["fator_potencia"] = 1.5  # fora do intervalo (0, 1]
    resp = client.post("/api/v1/faturas", json=dados)
    assert resp.status_code == 422


def test_duplicidade_de_competencia(client):
    client.post("/api/v1/faturas", json=_fatura_valida("2025-06"))
    resp = client.post("/api/v1/faturas", json=_fatura_valida("2025-06"))
    assert resp.status_code == 409


def test_obter_fatura_inexistente(client):
    resp = client.get("/api/v1/faturas/1999-12")
    assert resp.status_code == 404


def test_analise_consumo(client):
    client.post("/api/v1/faturas", json=_fatura_valida("2025-08"))
    resp = client.get("/api/v1/analises/consumo")
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["total_faturas"] >= 1
    assert corpo["consumo_total_kwh"] > 0
