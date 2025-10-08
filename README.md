This repository contains the code I used for my bachelor's degree. It propose an KAN-based hierarchical architecture for the prediction of financial assets' volatility.

## Application  
A basic **Python desktop application** is included in the repository.  
- Provides training, validation, inference, and visualization functionality.  
- Demonstrates the use of the **KAN-based models**, including the **hierarchical variants**.  
- Offers a reproducible experimentation environment for volatility forecasting.

## Data Availability  
- **Processed data**: already included in the repository for reproducibility.  
- **Raw data**: not distributed due to licensing restrictions, but can be downloaded from the original market data providers.  
- The code includes routines for downloading, processing, and preparing the raw data.  

# Using Kolmogorov–Arnold Networks for Predicting the Volatility of Financial Assets  

## Thesis abstract  
This thesis investigates **forecasting the volatility of financial assets** using **machine learning (ML) models** compared against statistical baselines. The dataset spans multiple asset classes—options, equity indices, FX pairs, and cryptocurrencies—and volatility is evaluated daily over two horizons: **1 day** and **20 days**. The main benchmark is **GARCH(1,1)** with Student-*t* residuals. The ML models include **MLP, LSTM, KAN, a recurrent KAN+LSTM hybrid, and iTransformer**.  

Findings are heterogeneous and depend on the asset, metric, and horizon. In the single-task setting, **LSTM and KAN** typically perform best on the 1-day horizon for non-equities (e.g., gold, EUR/USD), whereas for equities performance often declines, sometimes matching or underperforming the GARCH baseline. On the 20-day horizon, **GARCH frequently regains an advantage** on MSE/RMSE. Models capture association (Pearson correlation) and proportionality (QLIKE) reasonably well, but distance-based errors (MSE/RMSE/MAE) remain higher—especially for equities.  
Two enhancement directions were explored:  
1. **Multi-task learning** – Equal-weight multi-tasking had a neutral effect, while a hierarchical scheme—where intermediate outputs guide the main task—yielded gains for several symbols (e.g., EUR, gold, Apple).  
2. **Injecting GARCH parameters as inputs** – This generally degraded performance, contrary to some prior reports.  

## Conclusion  
ML models are competitive for volatility forecasting, but their advantage is **contingent on asset type, horizon, and evaluation metric**.  
Future research should:  
- Clarify the conditions and drivers of performance differences across assets and setups.  
- Refine **hierarchical multi-task strategies** for robust, longer-horizon forecasts.  
