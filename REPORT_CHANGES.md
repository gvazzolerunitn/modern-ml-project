# Report Integration - Notebook 1 & 2

## Summary of Changes

Successfully integrated both Short Notebook 1 and Short Notebook 2 into the comprehensive project report (`Report.tex`).

## Major Updates

### 1. **Abstract** (Updated)

- Now describes both complementary approaches
- Highlights Approach 1: MPE with material-specific models
- Highlights Approach 2: Ensemble with advanced features
- Mentions 53% validation improvement for Approach 2

### 2. **Section 4: Modeling Approach** (Enhanced)

- **4.3 Alternative Models Explored**: Added CatBoost comparison
  - XGBoost: 2-3× slower, comparable accuracy
  - **CatBoost: 1.5× slower, native quantile support, best for ensemble**
  - Updated comparison table with 3 models

### 3. **NEW Section 5: Advanced Ensemble Approach** (Added)

Complete description of Short Notebook 2:

#### 5.1 Enhanced Feature Engineering

- **34 additional advanced features** (51 total vs. 17 in Approach 1)
- **5.1.1 Advanced Temporal Features**:
  - Fourier features (weekly, monthly, quarterly seasonality)
  - Lag interaction features (lag × PO)
  - Higher-order rolling statistics (skewness, kurtosis, IQR)
  - Autocorrelation features (temporal dependency)
  - Trend and momentum features (acceleration)
- **5.1.2 Target Encoding**:
  - Material-level smoothed statistics
  - Leakage prevention (training set only)
  - 6 target-derived features
- **Table 5.1**: Complete feature breakdown (51 features categorized)

#### 5.2 Hyperparameter Optimization with Optuna

- Bayesian optimization (TPE algorithm)
- 100 trials per model
- Search space definitions (CatBoost & LightGBM)
- **Table 5.2**: Optimization results showing:
  - CatBoost: 14,707 → 9,295 (-37%)
  - LightGBM: 15,634 → 11,364 (-27%)

#### 5.3 Ensemble Strategy

- Weighted averaging formula
- **Table 5.3**: Multiple weight configurations (60/40, 65/35, 70/30)
- Global shrinkage factor (0.93-0.999)
- Rationale for CatBoost preference

#### 5.4 Training Strategy

- Temporal split: 2005-2023 train, 2024 validation
- 25,000 training samples, 5,000 validation samples
- Synthetic forecasting task generation

### 4. **Section 7: Results and Evaluation** (Completely Rewritten)

#### 7.1 Comparative Performance Analysis

Two separate subsections:

**7.1.1 Approach 1: Material-Specific Recursive Forecasting**

- **Table 7.1**: Complete performance summary
  - Validation QL: 20,017.90
  - 1,872 models trained
  - MAE objective + adaptive shrinkage
  - Computational cost: 45 min training, 8 min prediction
  - Key strengths listed

**7.1.2 Approach 2: Ensemble with Advanced Features**

- **Table 7.2**: Complete performance summary
  - CatBoost QL: 9,420
  - LightGBM QL: 11,550
  - Ensemble QL: ~10,200 (estimated)
  - 51 features (34 advanced)
  - Computational cost: 70 min total (60 min tuning + 10 min training)
  - Key strengths listed

#### 7.2 Kaggle Leaderboard Performance

- **Table 7.3**: Competition results for both approaches
- Submission file names for each approach

#### 7.3 Computational Efficiency

- **Table 7.4**: Side-by-side comparison
  - Runtime breakdown (feature eng, training, tuning, prediction)
  - Memory usage
  - Both approaches are production-ready (<12 hours)

### 5. **Section 8: Strengths, Limitations, and Future Improvements** (Expanded)

#### 8.1 Strengths

Three subsections:

- **Approach 1 strengths** (5 bullet points)
- **Approach 2 strengths** (6 bullet points)
- **Shared strengths** (4 bullet points)

#### 8.2 Limitations

Three subsections:

- **Approach 1 limitations** (4 bullet points)
- **Approach 2 limitations** (4 bullet points)
- **Shared limitations** (3 bullet points)

#### 8.3 Future Improvements

Four categories with specific suggestions:

- Model Architecture (hierarchical, LSTM, multi-quantile)
- Feature Engineering (PO integration, clustering, external signals)
- Calibration (learned shrinkage, conformal prediction)
- Validation Strategy (time series CV, stratification)

### 6. **Section 9: Conclusion** (Completely Rewritten)

- Emphasizes **two complementary systems**
- Separate paragraphs for each approach with key metrics
- **Key Contributions**: 7 bullet points
- Course requirements checklist (EDA, features, models, interpretation)
- Final statement emphasizing flexibility and production-readiness

## Statistics

### Document Size

- **Original**: ~784 lines
- **Updated**: ~1,200+ lines
- **Growth**: +53% (415+ lines added)

### New Content Added

- 1 new major section (Section 5: ~200 lines)
- 6 new tables
- 3 new equations
- 2 code listings (Optuna configurations)
- Complete rewrite of 3 sections (Results, Strengths/Limitations, Conclusion)

### Tables Added/Modified

1. **Table 4.1** (Modified): Model comparison (now includes CatBoost)
2. **Table 5.1** (New): Advanced feature categories (51 features)
3. **Table 5.2** (New): Optuna optimization results
4. **Table 5.3** (New): Ensemble weight configurations
5. **Table 7.1** (New): Approach 1 performance summary
6. **Table 7.2** (New): Approach 2 performance summary
7. **Table 7.3** (Modified): Kaggle performance (now 2 approaches)
8. **Table 7.4** (New): Computational efficiency comparison

## Key Improvements

### 1. Completeness

- Both notebooks now fully documented
- All major innovations explained
- Mathematical formulations provided
- Implementation details included

### 2. Technical Depth

- Advanced feature engineering fully described
- Hyperparameter optimization methodology explained
- Ensemble strategy rationale provided
- Temporal validation strategy documented

### 3. Comparison & Analysis

- Side-by-side performance metrics
- Computational cost comparison
- Strengths/weaknesses of each approach
- Use case recommendations

### 4. Professional Presentation

- Structured tables for easy comparison
- Clear section organization
- Comprehensive yet concise descriptions
- Academic writing style maintained

## What Each Approach Excels At

### Approach 1 (Material-Specific)

✅ **Best for**:

- Interpretability needs
- Material-level diagnostics
- Per-material feature importance
- Production deployment (faster inference)
- When computational resources are limited during inference

### Approach 2 (Ensemble)

✅ **Best for**:

- Maximum predictive accuracy
- Capturing complex patterns (seasonality, trends)
- When training time is not a constraint
- Leveraging multiple model types
- Direct quantile optimization

## Validation

- All LaTeX syntax follows proper formatting
- Tables properly structured with booktabs
- Equations numbered correctly
- Cross-references maintained
- Section hierarchy preserved

## Ready for Compilation

The report is ready to be compiled with `pdflatex` or any LaTeX distribution.
All dependencies are standard packages (already loaded in preamble).

---

**Last Updated**: 2025-11-08
**Authors**: Marco Prosperi, Andrea Richichi, Gianluigi Vazzoler
**Kaggle Team**: [66] AMG
