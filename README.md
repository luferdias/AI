# AI
Repositório de IA em Python com PyTorch/TensorFlow, Scikit-learn e XGBoost. Implementa ML clássico, CNNs e Transformers; pré-processamento, engenharia de features, pipelines de treino/validação, métricas e testes. Deploy via API, docs em Markdown e CI/CD com GitHub Actions.

## Recomendação de livros com embeddings

O script `src/recomendacao_livros.py` lê o arquivo `Base_livros.csv`, cria índices para usuários e livros, treina um modelo de recomendação com embeddings (user/item + dot product + camada de saída) e gera recomendações Top-N para um `ID_usuario` específico. Ele também salva o gráfico de `loss` por época em `loss.png`.

### Dependências

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### Execução

```bash
python src/recomendacao_livros.py --user-id 276725 --top-n 5 --epochs 10
```

### Interpretação do gráfico de loss

* **Tendência esperada:** a curva de treino deve diminuir ao longo das épocas; a validação deve acompanhar a queda.
* **Convergência:** quando as duas curvas estabilizam em valores próximos, o modelo tende a ter aprendido um padrão consistente.
* **Overfitting:** se a loss de treino continua caindo enquanto a de validação sobe, o modelo está memorizando demais.
* **Underfitting:** se ambas permanecem altas e pouco mudam, o modelo pode estar simples demais ou treinando pouco.

### Exemplo de recomendação (ilustrativo)

Para o usuário `276725`, o script retorna títulos com **maior score predito** (dot product + camada linear). Exemplo de saída (ilustrativa):

1. **Clara Callan** — maior score, indicando alta afinidade com o perfil do usuário.
2. **Decision in Normandy** — score levemente menor, mas ainda acima da média.
3. **Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It** — boa compatibilidade, porém abaixo dos anteriores.

O ranking é explicado pela **pontuação prevista** pelo modelo: quanto maior o score, maior a recomendação relativa daquele título para o usuário.
