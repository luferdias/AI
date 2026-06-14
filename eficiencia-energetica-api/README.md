# API de Eficiência Energética em Edifício Público

Trabalho final da disciplina **Construção de APIs para Inteligência Artificial** (UFG).

API em **FastAPI** que disponibiliza **dois serviços de IA com operações distintas**, aplicados a uma demanda real de trabalho: a análise de eficiência energética de edifícios públicos do MGI/SRA-ES (Grupo A, com geração fotovoltaica).

| # | Serviço de IA | Operação | Endpoint |
|---|---------------|----------|----------|
| 1 | **Classificação** da classe energética (A–E) | KNN — aprendizado supervisionado | `POST /api/v1/ia/classificar` |
| 2 | **Resumo executivo** do laudo | LLM (geração de texto), com fallback determinístico | `POST /api/v1/ia/resumir-analise` |

São operações genuinamente diferentes — classificação por modelo de ML e geração de texto por LLM — atendendo ao requisito de "pelo menos dois serviços que realizem operações diferentes".

## Sumário

- [Arquitetura](#arquitetura)
- [Requisitos](#requisitos)
- [Instalação e execução](#instalação-e-execução)
- [Autenticação](#autenticação)
- [Endpoints](#endpoints)
- [Exemplos de uso](#exemplos-de-uso)
- [Treino do modelo de IA](#treino-do-modelo-de-ia)
- [Testes](#testes)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Requisitos básicos atendidos](#requisitos-básicos-atendidos)

## Arquitetura

A API separa responsabilidades em camadas, de modo que a lógica de negócio (núcleo) seja testável de forma independente da camada web — princípio importante para auditabilidade em contexto público:

- **Camada web** (`routers/`): valida entrada, trata erros, registra logs e aplica autenticação.
- **Núcleo de domínio** (`features.py`, `diagnostico.py`, `ml/classificador.py`): funções puras, sem dependência do FastAPI.
- **Persistência** (`database.py`): repositório de faturas em memória (trocável por banco real sem afetar os routers).

O serviço de classificação segue o pipeline: dados brutos → `derivar_features` → vetor de 6 features → KNN → classe A–E. O serviço de resumo tenta o LLM e, na ausência de chave, recorre a um resumo por regra, garantindo execução sem dependências externas.

## Requisitos

- Python 3.12+
- (Opcional) [uv](https://docs.astral.sh/uv/) — alternativa ao pip
- (Opcional) Chave Groq para o serviço de LLM (há fallback sem ela)

## Instalação e execução

### Opção A — pip + venv

```bash
git clone <url-do-repositorio>
cd eficiencia-energetica-api

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # ajuste as variáveis se desejar

cd api
fastapi dev main.py
```

### Opção B — uv

```bash
uv sync
cp .env.example .env
cd api
uv run fastapi dev main.py
```

A API sobe em **http://localhost:8000**. A documentação interativa (Swagger UI) fica em **http://localhost:8000/docs**.

> O modelo de IA (`api/ml/modelo_knn.joblib`) já vem treinado e versionado: a API funciona imediatamente, sem etapa de treino.

## Autenticação

Os endpoints de IA e de diagnóstico são protegidos por **JWT**. Obtenha o token com o usuário de demonstração:

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=fiscal&password=fiscal123"
```

Resposta:

```json
{ "access_token": "eyJhbGci...", "token_type": "bearer" }
```

Use o token no cabeçalho `Authorization: Bearer <token>` das chamadas seguintes. No Swagger UI, clique em **Authorize** e informe usuário/senha.

## Endpoints

| Método | Rota | Autenticação | Descrição |
|--------|------|:---:|-----------|
| GET | `/health` | — | Verificação de saúde |
| POST | `/token` | — | Login; retorna o JWT |
| POST | `/api/v1/faturas` | — | Registra uma fatura mensal |
| GET | `/api/v1/faturas` | — | Lista as faturas |
| GET | `/api/v1/faturas/{mes_ano}` | — | Consulta uma fatura (404 se ausente) |
| GET | `/api/v1/analises/consumo` | — | Indicadores consolidados |
| POST | `/api/v1/diagnostico/demanda` | 🔒 | Diagnóstico de demanda (regra DR vs DC) |
| POST | `/api/v1/ia/classificar` | 🔒 | **IA 1** — classe energética A–E (KNN) |
| POST | `/api/v1/ia/resumir-analise` | 🔒 | **IA 2** — resumo executivo (LLM/fallback) |

## Exemplos de uso

### Serviço de IA 1 — Classificação (KNN)

```bash
curl -X POST http://localhost:8000/api/v1/ia/classificar \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "area_util_m2": 3000,
    "consumo_anual_kwh": 360000,
    "demanda_max_kw": 400,
    "horas_operacao_ano": 2900,
    "potencia_iluminacao_kw": 30,
    "potencia_climatizacao_kw": 120,
    "geracao_fv_anual_kwh": 90000,
    "ocupantes": 300
  }'
```

Resposta (resumida):

```json
{
  "classe": "C",
  "confianca": 0.73,
  "vetor_features": { "consumo_especifico": 120.0, "fator_carga": 0.31, "dpi": 10.0, "...": "..." },
  "probabilidades": { "A": 0.0, "B": 0.18, "C": 0.73, "D": 0.09, "E": 0.0 }
}
```

### Serviço de IA 2 — Resumo executivo

```bash
curl -X POST http://localhost:8000/api/v1/ia/resumir-analise \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "classe": "B",
    "situacao_demanda": "demanda_eficiente",
    "consumo_especifico": 120.0,
    "fracao_fv": 0.25,
    "fator_potencia": 0.993
  }'
```

O campo `origem` indica `"llm"` (gerado pelo modelo de linguagem) ou `"regra"` (fallback determinístico, quando não há `GROQ_API_KEY`).

## Treino do modelo de IA

O classificador KNN é treinado com `GridSearchCV` (busca de `k`, métrica de distância e ponderação) sobre um pipeline `StandardScaler → KNN`. Para regerar o modelo:

```bash
cd api
python ml/treinar_modelo.py
```

> **Sobre o dataset:** a planilha de faturas não traz rótulos de classe A–E. O treino usa um conjunto **sintético calibrado** a faixas plausíveis para edifícios públicos no clima de Vitória-ES, rotulado por uma **rubrica heurística transparente** (documentada em `treinar_modelo.py`). Não são valores normativos oficiais do PBE Edifica/INI-C — é uma heurística pedagógica adequada ao escopo do trabalho. Em produção, deve ser substituído por edifícios reais já etiquetados.

## Testes

```bash
pytest -q
```

A suíte cobre entradas válidas e inválidas, proteção por autenticação, a salvaguarda física do fator de carga e os dois serviços de IA.

## Estrutura do projeto

```
eficiencia-energetica-api/
├── api/
│   ├── main.py                 # app, registro de rotas e tratamento de erros
│   ├── models.py               # schemas Pydantic (contratos de dados)
│   ├── utils.py                # logging e ponte com o LLM (Groq)
│   ├── database.py             # repositório de faturas em memória
│   ├── features.py             # derivação do vetor de features
│   ├── diagnostico.py          # núcleo de regra DR vs DC
│   ├── ml/
│   │   ├── treinar_modelo.py   # geração do dataset + treino do KNN
│   │   ├── classificador.py    # inferência (carrega o modelo)
│   │   └── modelo_knn.joblib   # modelo treinado (versionado)
│   └── routers/
│       ├── health_router.py
│       ├── auth_router.py
│       ├── faturas_router.py
│       ├── diagnostico_router.py
│       └── ia_router.py
├── tests/
│   ├── conftest.py
│   ├── test_faturas.py
│   ├── test_diagnostico.py
│   └── test_ia.py
├── docs/
│   └── arquitetura.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Requisitos básicos atendidos

| Requisito | Como é atendido |
|-----------|-----------------|
| **Validação de dados** | Modelos Pydantic com restrições (intervalos, padrão de competência, enums) |
| **Tratamento de erros** | `HTTPException` (400, 404, 409) e handler padronizado para 422 |
| **Logs** | Módulo `logging` (`get_logger`) registra inclusões, erros e chamadas de IA |
| **Segurança** | Autenticação JWT (OAuth2) protegendo IA e diagnóstico |
| **Versionamento** | Prefixo de rota `/api/v1/...` |
| **Dois serviços de IA** | Classificação (KNN) e resumo (LLM) — operações distintas |
| **Execução por terceiros** | README, `requirements.txt`, `.env.example`, modelo versionado e testes |
