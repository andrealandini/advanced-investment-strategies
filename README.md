```markdown
# Advanced Investment Strategies

**Winter Semester 2025**  
University of Liechtenstein  

---

## Project Team
- **Andrea Landini** (240043) — [andrea.landini@uni.li](mailto:andrea.landini@uni.li)  
- **Simone Fruner** (240066) — [simone.fruner@uni.li](mailto:simone.fruner@uni.li)  
- **Ali Yaghoubi** (240085) — [ali.yaghoubi@uni.li](mailto:ali.yaghoubi@uni.li)  

**Promoter:** Dr. Gianluca De Nard — [gianluca.denard@uni.li](mailto:gianluca.denard@uni.li)  

---

## Project Overview

This project implements **VASA**, an advanced ensemble machine learning framework for financial forecasting and investment strategies. We explore two distinct approaches to quantifying market uncertainty:

1. **Volatility-based uncertainty** - Traditional approach using forward-looking standard deviation
2. **Entropy-based uncertainty** - Innovative approach using Shannon entropy of return distributions

---

## 📁 Project Structure

```
├── README.md                          # Project documentation (this file)
├── data/                              
│   ├── crsp_daily_top500.txt          # Raw CRSP daily data for top 500 stocks
│   ├── crsp_features_full_v.txt       # Feature dataset for volatility modeling
│   ├── entropy_target.txt             # Pre-computed entropy targets
│   ├── entropy-output/                # Results for entropy-based models
│   │   ├── ml_base_predictions_e.csv  # Base model predictions (entropy)
│   │   ├── vasa_metrics_e.csv         # VASA performance metrics (entropy)
│   │   └── vasa_r2_e.csv              # R² results (entropy)
│   └── vol-output/                    # Results for volatility-based models  
│       ├── ml_base_predictions_v.csv  # Base model predictions (volatility)
│       ├── vasa_predictions_v2.csv    # Final VASA predictions
│       └── results_v2/                # Investment strategy results
│           ├── monthly_portfolios_vasa_10y.csv  # Portfolio compositions
│           └── monthly_returns_vasa_10y.csv     # Strategy performance
├── projects/
│   ├── ais_paper_AL-SF-AY.pdf         # Final research paper
│   └── vasa_final_AL.pptx             # Presentation slides
├── references/                        # Academic literature
├── src/
│   ├── method.Rmd                     # Main methodology documentation
│   └── vasa_monthly_strategy.r        # Monthly rebalancing implementation
└── utils/                             # Core implementation scripts
    ├── build_entropy_target.r         # Creates entropy-based uncertainty measure
    ├── train_models_[e/v].R           # Trains base ML models (entropy/volatility)
    ├── vasa_[e/v].R                   # Implements VASA ensemble (entropy/volatility)
    ├── cross_vasa_[e/v].R             # Cross-sectional VASA implementation
    └── vasa_v2.R                      # Enhanced VASA with improved metrics
```

---

## Methodology

### 1. Feature Engineering
- **Return Lags**: Historical returns (1-5 days)
- **Rolling Statistics**: 20-day mean, standard deviation, skewness
- **Uncertainty Targets**:
  - *Volatility*: Forward 20-day standard deviation
  - *Entropy*: Forward 20-day Shannon entropy of return distribution

### 2. Machine Learning Pipeline
- **Base Models**: Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, Neural Networks
- **VASA Ensemble**: 
  - Randomly subsample base models (k=3 from 6 available)
  - Train Ridge regression on each subsample
  - Average predictions across B=100 subsamples
  - Reduces overfitting and improves robustness

### 3. Investment Strategies
- **Inverse Volatility**: Long-only portfolio weighted by low volatility
- **Low-High Volatility**: Long/short strategy going long low-vol and short high-vol stocks
- **Target Volatility Overlay**: Dynamic position sizing based on market volatility
- **Monthly Low-Vol Rotation**: Systematic rebalancing into 10 lowest-volatility stocks

---

##  How to Reproduce

### 1. Data Preparation
```r
# Build entropy targets
source("utils/build_entropy_target.r")

# Generate feature sets
source("src/method.Rmd")  # Contains feature engineering code
```

### 2. Model Training
```r
# Train base models for volatility
source("utils/train_models_v.R")

# Train base models for entropy  
source("utils/train_models_e.R")
```

### 3. VASA Implementation
```r
# Run VASA ensemble
source("utils/vasa_v2.R")        # Enhanced volatility VASA
source("utils/vasa_e.R")         # Entropy VASA
source("utils/cross_vasa_v.R")   # Cross-sectional analysis
```

### 4. Strategy Implementation
```r
# Execute investment strategies
source("src/vasa_monthly_strategy.r")
```

---

## Results Interpretation

The project demonstrates that:
- **Machine learning ensembles** improve financial forecasting accuracy
- **Subsampling aggregation** (VASA) effectively reduces overfitting in high-dimensional finance problems
- **Alternative uncertainty measures** (entropy) offer new perspectives on risk modeling
- **Systematic low-volatility strategies** remain profitable in modern markets

---

## References

- De Nard, Gianluca; Hediger, Simon; Leippold, Markus (2022). *Subsampled factor models for asset pricing: The rise of Vasa*. Journal of Forecasting, 41(6), 1217–1247. [Wiley Online Library](https://onlinelibrary.wiley.com/journal/1099131x)  
- Gu, Shihao; Kelly, Bryan; Xiu, Dacheng (2020). *Empirical asset pricing via machine learning*. The Review of Financial Studies, 33(5), 2223–2273. [Oxford University Press](https://academic.oup.com/rfs)  

---

## Future Work

- Incorporate additional alternative data sources
- Extend to international markets and asset classes
- Develop real-time trading implementation
- Explore deep learning architectures for uncertainty estimation

---

*Last updated: Winter Semester 2025*
```
