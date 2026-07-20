import nbformat as nbf
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('# FinSight - Enterprise ML & Credit Risk Prediction Engine\n\nProduction-ready ML pipeline for loan default prediction.'),

    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import os, sys, time, warnings, json
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('..'))

from src.ml.preprocessor import MLPreprocessor
from src.ml.trainer import MLTrainer
from src.ml.evaluator import MLEvaluator
from src.ml.predictor import MLPredictor
from src.analytics.report_generator import ReportGenerator

# ============================================================
# CONFIGURATION - PRODUCTION PARAMETERS
# ============================================================
BASELINE_SAMPLE_SIZE = 50_000   # Change this to adjust benchmark sample
RANDOM_SEED = 42
OPTUNA_TRIALS = 15
np.random.seed(RANDOM_SEED)
print('Modules loaded. Configuration set.')'''),

    nbf.v4.new_markdown_cell('## 1. Load Segmented Dataset'),
    nbf.v4.new_code_cell('''DATA_PATH = '../data/processed/application_train_segmented.csv'
df = pd.read_csv(DATA_PATH)
print(f'Loaded dataset: {df.shape}')
print(f'Target Distribution:\\n{df["TARGET"].value_counts(normalize=True)}')'''),

    nbf.v4.new_markdown_cell('## 2. Data Preparation (Stratified 80/10/10 Split)'),
    nbf.v4.new_code_cell('''preprocessor = MLPreprocessor(target_col='TARGET')
X, y = preprocessor.prepare_data(df)

# Fill remaining NaN for ML compatibility
X = X.fillna(0)
print(f'Feature matrix: {X.shape}, Target: {y.shape}')

splits = preprocessor.split_data(X, y)
X_train_full, y_train_full = splits['X_train'], splits['y_train']
X_val, y_val = splits['X_val'], splits['y_val']
X_test, y_test = splits['X_test'], splits['y_test']

print(f'Train: {X_train_full.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
print(f'Train Target dist: {y_train_full.value_counts(normalize=True).to_dict()}')'''),

    nbf.v4.new_markdown_cell('## 3. Baseline Benchmarking on Stratified Sample'),
    nbf.v4.new_code_cell('''# Stratified sample for benchmarking
from sklearn.model_selection import train_test_split

if len(X_train_full) > BASELINE_SAMPLE_SIZE:
    X_sample, _, y_sample, _ = train_test_split(
        X_train_full, y_train_full,
        train_size=BASELINE_SAMPLE_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_SEED
    )
    print(f'Using stratified sample of {BASELINE_SAMPLE_SIZE} for benchmarking.')
else:
    X_sample, y_sample = X_train_full, y_train_full
    print(f'Dataset fits within sample size. Using full training set.')

trainer = MLTrainer(model_dir='../models/trained/')
comparison_table = trainer.benchmark_models(X_sample, y_sample, X_val, y_val)
print('\\n=== MODEL COMPARISON TABLE ===')
print(comparison_table.to_string(index=False))'''),

    nbf.v4.new_markdown_cell('## 4. Automatic Top Model Selection (Data-Driven)'),
    nbf.v4.new_code_cell('''top_models = trainer.evaluator.select_top_n_models(n=2)
print(f'Top 2 models selected for hyperparameter optimization: {top_models}')'''),

    nbf.v4.new_markdown_cell('## 5. Hyperparameter Optimization (Optuna)'),
    nbf.v4.new_code_cell('''optimized_models = {}
for model_name in top_models:
    print(f'\\nOptimizing {model_name}...')
    optimized = trainer.optimize_hyperparameters(
        model_name, X_sample, y_sample, X_val, y_val, n_trials=OPTUNA_TRIALS
    )
    optimized_models[model_name] = optimized
    
print(f'\\nOptimization complete for: {list(optimized_models.keys())}')'''),

    nbf.v4.new_markdown_cell('## 6. Final Model Training on FULL Production Dataset'),
    nbf.v4.new_code_cell('''# Select the champion (first in top_models list)
champion_name = top_models[0]
champion_model = optimized_models[champion_name]

# Train on FULL training data (not the sample)
trainer.train_final_model(champion_model, champion_name, X_train_full, y_train_full)
print(f'Champion model ({champion_name}) trained on full production dataset.')'''),

    nbf.v4.new_markdown_cell('## 7. Final Evaluation on Held-Out Test Set'),
    nbf.v4.new_code_cell('''from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

y_pred_test = champion_model.predict(X_test)
y_prob_test = champion_model.predict_proba(X_test)[:, 1] if hasattr(champion_model, 'predict_proba') else y_pred_test

print('=== FINAL TEST SET EVALUATION ===')
print(classification_report(y_test, y_pred_test, target_names=['Repay', 'Default']))
print(f'Confusion Matrix:\\n{confusion_matrix(y_test, y_pred_test)}')
print(f'\\nROC-AUC: {roc_auc_score(y_test, y_prob_test):.4f}')
print(f'PR-AUC: {average_precision_score(y_test, y_prob_test):.4f}')'''),

    nbf.v4.new_markdown_cell('## 8. Feature Importance'),
    nbf.v4.new_code_cell('''# Extract feature importance if available
if hasattr(champion_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': X_train_full.columns,
        'Importance': champion_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    importances.to_csv('../reports/Feature_Importance.csv', index=False)
    print('Top 20 Features:')
    print(importances.head(20).to_string(index=False))
else:
    print('Model does not expose feature_importances_.')'''),

    nbf.v4.new_markdown_cell('## 9. Generate Reports & Save Outputs'),
    nbf.v4.new_code_cell('''report_gen = ReportGenerator(output_dir='../reports/')

# Model Comparison Report
comp_lines = ['Model Comparison Results (Ranked by Weighted Score: 40% PR-AUC, 30% Recall, 20% ROC-AUC, 10% F1)', '']
for _, row in comparison_table.iterrows():
    comp_lines.append(f"{row['Ranking']}. {row['Model']} - PR-AUC: {row['PR-AUC']:.4f}, Recall: {row['Recall']:.4f}, ROC-AUC: {row['ROC-AUC']:.4f} [{row['Business Recommendation']}]")
report_gen.generate_pdf_report('FinSight Model Comparison Report', comp_lines, 'Model_Comparison_Report.pdf')

# Model Performance Report
perf_lines = [
    f'Champion Model: {champion_name}',
    f'ROC-AUC (Test): {roc_auc_score(y_test, y_prob_test):.4f}',
    f'PR-AUC (Test): {average_precision_score(y_test, y_prob_test):.4f}',
    f'Trained on {len(X_train_full)} samples.',
    f'Evaluated on {len(X_test)} hold-out samples.'
]
report_gen.generate_pdf_report('FinSight Model Performance Report', perf_lines, 'Model_Performance_Report.pdf')

# Training Report Markdown
with open('../reports/Training_Report.md', 'w') as f:
    f.write(f'# ML Training Report\\n\\n')
    f.write(f'**Champion Model**: {champion_name}\\n\\n')
    f.write(f'**Baseline Sample Size**: {BASELINE_SAMPLE_SIZE}\\n\\n')
    f.write(f'**Optuna Trials**: {OPTUNA_TRIALS}\\n\\n')
    f.write(f'**Final Training Size**: {len(X_train_full)}\\n\\n')
    f.write(comparison_table.to_markdown(index=False))

# Predictions Sample
predictor = MLPredictor(model_dir='../models/trained/')
sample_df = X_test.head(100).copy()
sample_preds = predictor.batch_predict(sample_df)
sample_preds.to_csv('../reports/Predictions_Sample.csv', index=False)

# Save comparison table CSV
comparison_table.to_csv('../reports/Model_Comparison_Table.csv', index=False)

print('All reports and outputs generated successfully.')''')
]

nb['cells'] = cells
with open('notebooks/04_Machine_Learning.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook created successfully.')
