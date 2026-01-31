# CLAUDE.md - AI Repository Guide

This document provides essential context for AI assistants working with this codebase.

## Project Overview

This is an educational AI/Machine Learning repository for the "Frameworks de IA" (FRA) specialization course at UFPR/SEPT. The project implements practical ML pipelines focusing on:

- **Recommendation Systems** - Collaborative filtering with neural embeddings
- **Neural Network Classification** - Fashion MNIST image classification
- **Regression Tasks** - Wine quality prediction
- **Deep Learning Visualization** - DeepDream algorithm with InceptionV3
- **Feature Engineering** - Feature selection techniques for predictive modeling

**Primary Language**: Python with Portuguese documentation and variable naming

## Directory Structure

```
/home/user/AI/
├── src/                          # Production Python scripts
│   └── recomendacao_livros.py    # Book recommendation CLI tool
├── *.ipynb                       # Jupyter notebooks (exercises/tutorials)
├── *.pdf                         # Course materials from Aula 22-23
├── Base_livros.csv               # Book ratings dataset (~100K+ records)
├── README.md                     # Project overview (Portuguese)
└── CLAUDE.md                     # This file
```

## Key Files

| File | Purpose |
|------|---------|
| `src/recomendacao_livros.py` | Main CLI script for book recommendations with embeddings |
| `EX2.ipynb` | Fashion MNIST classification with neural networks |
| `EX3_Regressao_Wine_Quality.ipynb` | Wine quality regression |
| `EX3_Sistema_Recomendacao_Livros.ipynb` | Book recommendation system (Aula 22) |
| `Feature_Selection_Airline_Satisfaction.ipynb` | Feature selection techniques |
| `Deepdream_Felino.ipynb` | Minimal DeepDream implementation |
| `FRA - Aula 23 - Prática DeepDream.ipynb` | Complete DeepDream tutorial |

## Technology Stack

### Core Frameworks
- **TensorFlow/Keras** - Primary deep learning framework
- **Pandas** - Data manipulation and CSV handling
- **NumPy** - Numerical computations
- **Scikit-learn** - Feature selection, preprocessing
- **Matplotlib** - Visualization and plotting
- **PIL/Pillow** - Image processing (DeepDream)

### Environment Setup
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### Environment Variables
```bash
TF_USE_LEGACY_KERAS=1  # Required for legacy Keras compatibility
```

## Code Conventions

### Naming
- **Portuguese naming** for variables, functions, columns, and documentation
- Descriptive function names: `load_data()`, `prepare_mappings()`, `build_model()`, `train_model()`
- Dataset columns: `ID_usuario`, `Titulo`, `Notas`, `ISBN`

### Type Hints
- Use Python 3.10+ annotations: `from __future__ import annotations`
- Return type hints on all functions: `def function() -> ReturnType:`
- Parameter type hints: `def function(param: Type) -> ReturnType:`

### Code Organization Pattern
```python
# 1. Imports (standard library, then third-party)
# 2. Environment configuration
# 3. Data loading functions
# 4. Data preparation functions
# 5. Model building functions
# 6. Training functions
# 7. Inference/prediction functions
# 8. Visualization functions
# 9. Main entry point with argparse
```

### Neural Network Architecture Pattern
```
Input → Embedding → Flatten → Concatenate → Dense(1024, ReLU) → Dense(1, Linear) → Output
```

### Training Conventions
- Center ratings by training mean for normalization
- Default 80/20 train/validation split
- Use SGD optimizer with momentum (learning_rate=0.08, momentum=0.9)
- MSE loss for regression tasks
- Batch processing for large-scale predictions

### Error Handling
- Validate required columns exist in datasets
- Check for user existence before recommendations
- Handle edge cases: empty results, missing data

## Running the Code

### Book Recommendation Script
```bash
python src/recomendacao_livros.py --user-id 276729 --top-n 5 --epochs 15
```

**CLI Arguments:**
- `--csv`: Path to CSV file (default: `Base_livros.csv`)
- `--user-id`: Target user ID for recommendations (required)
- `--top-n`: Number of recommendations (default: 5)
- `--epochs`: Training epochs (default: 25)
- `--embedding-dim`: Embedding dimension (default: 10)
- `--batch-size`: Training batch size (default: 1024)
- `--val-split`: Validation split ratio (default: 0.2)
- `--plot`: Loss plot output path (default: `loss.png`)

### Jupyter Notebooks
Run with Jupyter Lab or Google Colab. Notebooks are self-contained with inline documentation.

## Git Workflow

### Branch Naming
- Feature branches: `claude/description-sessionid` or `codex/task-name`
- Merge via Pull Requests

### Commit Messages
- Use descriptive messages explaining the change
- Reference course materials (Aula 22, Aula 23) when implementing course exercises

## Data Files

### Base_livros.csv
- ~100K+ book rating records
- Columns: `ISBN`, `Titulo`, `Autor`, `Ano`, `Editora`, `ID_usuario`, `Notas`
- Used for training recommendation models

## Guidelines for AI Assistants

### When Modifying Code
1. Preserve Portuguese naming conventions
2. Maintain type hints on all functions
3. Follow the existing modular function decomposition
4. Keep code aligned with course material architecture when specified
5. Use relative imports within the `src/` directory

### When Adding Notebooks
1. Include clear section headers in Portuguese
2. Add markdown explanations before code cells
3. Reference relevant course materials (Aula number)
4. Include visualization outputs where appropriate

### When Working with Data
1. Validate required columns before processing
2. Handle categorical encoding with `pd.Categorical`
3. Center/normalize ratings when building models
4. Use batch processing for large-scale predictions

### Common Tasks
- **Add new ML exercise**: Create notebook following `EX*.ipynb` pattern
- **Improve recommendation model**: Modify `src/recomendacao_livros.py`
- **Add visualization**: Use matplotlib with Portuguese labels
- **Update dependencies**: Document in README.md (no requirements.txt)

## Course Material References

- **Aula 22 (4.3)**: Book recommendation system with embeddings
- **Aula 23**: DeepDream visualization with InceptionV3

PDF materials in the repository provide architectural guidance for implementations.
