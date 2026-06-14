# Decisões de arquitetura

## Separação núcleo / web

Toda a lógica de domínio é implementada como **funções puras**, isoladas do FastAPI:

- `features.derivar_features` — transforma dados brutos em vetor de features;
- `diagnostico.diagnosticar` — aplica a regra DR vs DC;
- `ml.classificador.classificar_vetor` — executa a inferência do KNN.

Os routers apenas validam entrada, tratam erros, registram logs e chamam o núcleo. Essa fronteira permite testar a regra de negócio sem subir o servidor e mantém a trilha de auditoria exigida em contexto público: cada serviço pode ser verificado isoladamente.

## Por que orquestração desacoplada (Pattern B)

Os serviços de classificação e diagnóstico são endpoints independentes, não um pipeline encadeado. Em fiscalização, a auditabilidade de cada decisão isolada vale mais que a conveniência de uma única chamada: é possível reclassificar sem rediagnosticar, e cada resultado é registrado com sua entrada e saída.

## Os dois serviços de IA

```
                         ┌─────────────────────────────────────────┐
  dados brutos  ─────────▶  IA 1: /ia/classificar                   │
  (área, consumo,        │     derivar_features → KNN → classe A–E   │
   demanda, FV, …)       └─────────────────────────────────────────┘

  indicadores   ─────────┌─────────────────────────────────────────┐
  (classe, demanda,      │  IA 2: /ia/resumir-analise                │
   consumo esp., FV,     │     LLM (Groq) ──┐                        │
   fator de potência)    │                  ├─▶ resumo executivo     │
                         │     fallback ────┘   (origem: llm/regra)  │
                         └─────────────────────────────────────────┘
```

**Operações distintas:** a IA 1 é classificação (modelo de ML supervisionado); a IA 2 é geração de texto (LLM). O fallback determinístico da IA 2 garante reprodutibilidade pela banca mesmo sem chave de LLM.

## Pipeline do classificador

`StandardScaler → KNeighborsClassifier`, com hiperparâmetros (`n_neighbors`, `weights`, métrica `p`) selecionados por `GridSearchCV` (validação cruzada 5-fold). A padronização é essencial: sem ela, o consumo específico (ordem de centenas) dominaria a distância sobre features de ordem unitária como o fator de carga.

## Salvaguarda física

A derivação de features recusa entradas fisicamente impossíveis (fator de carga > 1, em que a demanda média excederia a máxima registrada), retornando HTTP 400 com a explicação — em vez de classificar sobre dado inconsistente.
