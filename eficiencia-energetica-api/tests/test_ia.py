"""Testes dos serviços de IA: classificação (KNN) e resumo executivo (LLM/fallback)."""


def _features_validas():
    return {
        "area_util_m2": 3000.0,
        "consumo_anual_kwh": 360000.0,
        "demanda_max_kw": 400.0,
        "horas_operacao_ano": 2900.0,
        "potencia_iluminacao_kw": 30.0,
        "potencia_climatizacao_kw": 120.0,
        "geracao_fv_anual_kwh": 90000.0,
        "ocupantes": 300,
    }


def test_classificar_exige_autenticacao(client):
    resp = client.post("/api/v1/ia/classificar", json=_features_validas())
    assert resp.status_code == 401


def test_classificar_valido(client, auth_header):
    resp = client.post("/api/v1/ia/classificar", json=_features_validas(), headers=auth_header)
    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["classe"] in {"A", "B", "C", "D", "E"}
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert "consumo_especifico" in corpo["vetor_features"]


def test_classificar_salvaguarda_fator_carga(client, auth_header):
    # Consumo/horas gera demanda média > demanda máxima -> fator de carga > 1.
    dados = _features_validas()
    dados.update({"consumo_anual_kwh": 900000.0, "horas_operacao_ano": 1000.0, "demanda_max_kw": 100.0})
    resp = client.post("/api/v1/ia/classificar", json=dados, headers=auth_header)
    assert resp.status_code == 400


def test_classificar_entrada_invalida(client, auth_header):
    dados = _features_validas()
    dados["area_util_m2"] = -1
    resp = client.post("/api/v1/ia/classificar", json=dados, headers=auth_header)
    assert resp.status_code == 422


def test_resumir_analise(client, auth_header):
    payload = {
        "classe": "B",
        "situacao_demanda": "demanda_eficiente",
        "consumo_especifico": 120.0,
        "fracao_fv": 0.25,
        "fator_potencia": 0.993,
    }
    resp = client.post("/api/v1/ia/resumir-analise", json=payload, headers=auth_header)
    assert resp.status_code == 200
    corpo = resp.json()
    assert len(corpo["resumo"]) > 50
    assert corpo["origem"] in {"llm", "regra"}
