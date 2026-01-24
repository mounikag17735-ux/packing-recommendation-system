# EcoPack AI — Sustainable Packaging Recommendation System
🔗 Live App: https://huggingface.co/spaces/mounikag17735/ecopack-ai

EcoPack AI is a Machine Learning powered Flask web application that recommends the most cost-efficient and eco-friendly packaging material based on product characteristics like weight, fragility, industry, and sustainability priority.

## What This Project Demonstrates
- Flask web application development
- End-to-end Machine Learning pipeline
- Cloud database integration using SQLiteCloud
- Live analytics dashboard with charts
- Excel & PDF report generation
- Deployment on HuggingFace Spaces (Docker)

## How It Works
- User enters product details
- ML models predict best packaging material
- Recommendation is logged to cloud DB
- Dashboard visualizes sustainability metrics

## ML Models
### Model	                                         Purpose
Random Forest	                                     Cost Prediction
XGBoost	CO₂                                        Impact Prediction

## Dashboard

- Average CO₂ score by material

- Average cost score by material

- Material usage frequency

## Tech Stack

Python • Flask • Pandas • Scikit-learn • XGBoost • SQLiteCloud • Matplotlib • ReportLab • OpenPyXL

## Run Locally
pip install -r requirements.txt
python run.py

