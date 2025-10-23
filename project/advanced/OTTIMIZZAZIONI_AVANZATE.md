# 🎨 OTTIMIZZAZIONI AVANZATE (Opzionali)

## 📖 Questo file contiene suggerimenti per migliorare ulteriormente i risultati

---

## 1️⃣ Hyperparameter Tuning con Optuna

### Aggiungi questa sezione in `advanced_modeling.ipynb`:

```python
import optuna

def objective_lgb(trial):
    \"\"\"Objective function per LightGBM\"\"\"
    params = {
        'objective': 'quantile',
        'alpha': 0.8,
        'num_leaves': trial.suggest_int('num_leaves', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    
    # Training con cross-validation
    cv_results = lgb.cv(
        params,
        train_data_lgb,
        num_boost_round=1000,
        nfold=5,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return cv_results['valid quantile-mean'][-1]

# Ottimizzazione
study = optuna.create_study(direction='minimize')
study.optimize(objective_lgb, n_trials=50, timeout=3600)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value}")
```

**Tempo**: ~1-2 ore  
**Miglioramento atteso**: 2-5%

---

## 2️⃣ Feature Selection con SHAP

### Rimuovi feature ridondanti:

```python
import shap

# Calcola SHAP values
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'shap_importance': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_importance', ascending=False)

# Seleziona top N feature
top_n = 100  # invece di 150+
selected_features = feature_importance.head(top_n)['feature'].tolist()

print(f"Selected {len(selected_features)} most important features")

# Re-train con feature selezionate
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
```

**Benefici**:
- Training più veloce
- Meno overfitting
- Modello più interpretabile

---

## 3️⃣ Stacking al Posto di Weighted Average

### Usa un meta-learner:

```python
from sklearn.linear_model import Ridge

# Predizioni out-of-fold dei modelli base
oof_predictions = np.column_stack([
    lgb_pred_train,
    catboost_pred_train,
    xgb_pred_train,
    nn_pred_train
])

# Meta-learner (Ridge con quantile loss approssimato)
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_predictions, y_train)

# Predizioni finali
test_predictions = np.column_stack([
    lgb_pred_test,
    catboost_pred_test,
    xgb_pred_test,
    nn_pred_test
])

stacked_predictions = meta_model.predict(test_predictions)

# Evaluate
stacking_metrics = evaluate_model(y_test, stacked_predictions, "Stacking")
```

**Miglioramento atteso**: 1-3%

---

## 4️⃣ Post-Processing delle Predizioni

### Smoothing temporale:

```python
# Nel notebook generate_predictions.ipynb

# Dopo aver generato le predizioni
submission_with_dates = submission.merge(
    prediction_mapping[['ID', 'rm_id', 'forecast_end_date']], 
    on='ID'
)

# Smoothing per rm_id
submission_with_dates['predicted_weight_smoothed'] = (
    submission_with_dates
    .sort_values(['rm_id', 'forecast_end_date'])
    .groupby('rm_id')['predicted_weight']
    .transform(lambda x: x.rolling(window=3, min_periods=1, center=True).mean())
)

submission['predicted_weight'] = submission_with_dates['predicted_weight_smoothed']
```

**Benefici**: Predizioni più stabili temporalmente

---

## 5️⃣ Ensemble con Modelli Temporali

### Aggiungi un modello LSTM:

```python
from torch import nn
import torch.nn.functional as F

class LSTMQuantile(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(LSTMQuantile, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Prendi l'ultimo output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze()

# Prepara sequenze temporali
def create_sequences(data, seq_len=7):
    sequences = []
    for rm_id in data['rm_id'].unique():
        rm_data = data[data['rm_id'] == rm_id].sort_values('date_arrival')
        for i in range(len(rm_data) - seq_len):
            seq = rm_data.iloc[i:i+seq_len][feature_cols].values
            target = rm_data.iloc[i+seq_len]['net_weight']
            sequences.append((seq, target))
    return sequences

# Training LSTM...
```

**Miglioramento atteso**: 3-7% (se ci sono forti pattern temporali)

---

## 6️⃣ Feature Engineering Aggiuntivo

### Aggiungi queste feature in `advanced_feature_engineering.ipynb`:

```python
# 1. Interazioni di secondo ordine
data['supplier_rm_product'] = (
    data['supplier_id'].astype(str) + '_' + 
    data['rm_id'].astype(str) + '_' + 
    data['product_id'].astype(str)
)

# 2. Differenze percentuali
data['pct_change_lag1'] = data.groupby('rm_id')['net_weight'].pct_change()
data['pct_change_lag7'] = data.groupby('rm_id')['net_weight'].pct_change(periods=7)

# 3. Exponential moving average
data['ema_7'] = data.groupby('rm_id')['net_weight'].transform(
    lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
)
data['ema_30'] = data.groupby('rm_id')['net_weight'].transform(
    lambda x: x.ewm(span=30, adjust=False).mean().shift(1)
)

# 4. Volatilità
data['volatility_7'] = data.groupby('rm_id')['net_weight'].transform(
    lambda x: x.rolling(7).std() / (x.rolling(7).mean() + 1e-6)
).shift(1)

# 5. Z-score (outlier detection)
data['zscore'] = data.groupby('rm_id')['net_weight'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)

# 6. Rank features
data['rank_in_supplier'] = data.groupby('supplier_id')['net_weight'].rank(pct=True)
data['rank_in_rm'] = data.groupby('rm_id')['net_weight'].rank(pct=True)
```

---

## 7️⃣ Cross-Validation più Sofisticata

### Time Series Cross-Validation:

```python
from sklearn.model_selection import TimeSeriesSplit

# Invece di un singolo split 80/20
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"\\nFold {fold+1}")
    
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    # Train model
    fold_model = lgb.LGBMRegressor(**lgb_params)
    fold_model.fit(X_fold_train, y_fold_train)
    
    # Evaluate
    fold_preds = fold_model.predict(X_fold_val)
    fold_score = quantile_loss(y_fold_val, fold_preds, 0.8)
    cv_scores.append(fold_score)
    
    print(f"Fold {fold+1} Quantile Loss: {fold_score:.2f}")

print(f"\\nCV Mean: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

---

## 8️⃣ Calibrazione delle Predizioni

### Adjusting per bias:

```python
# Dopo aver generato le predizioni sul test set
# Calcola bias storico per rm_id

rm_bias = {}
for rm_id in historical_data['rm_id'].unique():
    rm_hist = historical_data[historical_data['rm_id'] == rm_id]
    if len(rm_hist) > 10:
        # Calcola tendenza: crescita o decrescita
        recent = rm_hist.tail(30)['net_weight'].mean()
        old = rm_hist.head(30)['net_weight'].mean()
        bias = (recent - old) / (old + 1e-6)
        rm_bias[rm_id] = bias

# Applica bias alle predizioni
submission['rm_id'] = pred_data['rm_id']
submission['bias'] = submission['rm_id'].map(rm_bias).fillna(0)
submission['predicted_weight_adjusted'] = (
    submission['predicted_weight'] * (1 + submission['bias'] * 0.1)  # Fattore 0.1 da tuning
)

# Usa adjusted come predizioni finali
submission['predicted_weight'] = submission['predicted_weight_adjusted']
```

---

## 9️⃣ Ensemble di Ensemble

### Multi-level stacking:

```python
# Level 1: Base models (già hai)
base_models = [lgb_model, catboost_model, xgb_model, nn_model]

# Level 2: Meta-models con diverse configurazioni
meta_models = [
    Ridge(alpha=0.1),
    Ridge(alpha=1.0),
    Ridge(alpha=10.0),
    Lasso(alpha=0.1),
]

# Level 3: Final combiner
# ... training code ...
```

**Nota**: Molto più complesso, usa solo se hai tempo

---

## 🔟 Augmented Features da External Data

### Se hai accesso a dati esterni:

```python
# Esempio: dati meteo, festività, eventi economici
# (non disponibili in questa competition, ma utile per future)

# 1. Festività
import holidays
norway_holidays = holidays.Norway()

data['is_holiday'] = data['date_arrival'].apply(
    lambda x: 1 if x in norway_holidays else 0
)
data['days_to_holiday'] = ... # calcola distanza dal prossimo festivo

# 2. Stagionalità business
data['is_end_of_quarter'] = data['date_arrival'].dt.is_quarter_end.astype(int)
data['is_end_of_year'] = (data['date_arrival'].dt.month == 12).astype(int)
```

---

## 🎯 Priorità Ottimizzazioni

### Se hai tempo limitato, fai in ordine:

1. **Feature Selection (SHAP)** - Quick win, sempre utile
2. **Post-Processing** - Facile da implementare
3. **Hyperparameter Tuning** - Miglioramento garantito
4. **Additional Features** - Se hai idee specifiche
5. **Stacking** - Se vuoi spingere ulteriormente
6. **LSTM** - Solo se hai molto tempo

---

## 📊 Tracking Esperimenti

### Usa MLflow o simili:

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(lgb_params)
    
    # Log metrics
    mlflow.log_metric("quantile_loss", lgb_metrics_test['quantile_loss'])
    mlflow.log_metric("mae", lgb_metrics_test['mae'])
    
    # Log model
    mlflow.lightgbm.log_model(lgb_model, "model")
```

---

## 💡 Tips Finali

### Best Practices:
- ✅ Salva ogni esperimento
- ✅ Usa git per versioning
- ✅ Documenta cosa hai provato
- ✅ Fai una modifica alla volta
- ✅ Valida sempre su test set

### Quando fermarsi:
- ⚠️ Se il miglioramento < 0.5% → probabilmente overfitting
- ⚠️ Se il train loss cala ma test loss sale → overfitting
- ⚠️ Se hai raggiunto il tempo limite → usa ciò che hai

---

## 📈 Roadmap Miglioramenti

### Week 1:
- [ ] Implementa SHAP feature selection
- [ ] Hyperparameter tuning base (Optuna)
- [ ] Post-processing smoothing

### Week 2:
- [ ] Aggiungi nuove feature (EMA, volatility)
- [ ] Stacking ensemble
- [ ] Cross-validation 5-fold

### Week 3:
- [ ] LSTM model (se necessario)
- [ ] Calibrazione predizioni
- [ ] Fine-tuning finale

---

## 🎓 Risorse Aggiuntive

### Kaggle:
- [Quantile Regression Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Feature Engineering](https://www.kaggle.com/learn/feature-engineering)
- [Ensemble Methods](https://www.kaggle.com/learn/intermediate-machine-learning)

### Papers:
- "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
- "XGBoost: A Scalable Tree Boosting System"
- "CatBoost: unbiased boosting with categorical features"

---

## ⚠️ Warning

**Non implementare tutto insieme!**

- Fai una modifica alla volta
- Valuta l'impatto
- Tieni ciò che funziona
- Scarta ciò che peggiora

**Remember**: Simple is better than complex!

---

**Buon divertimento con le ottimizzazioni! 🚀**
