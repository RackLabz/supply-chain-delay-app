# Supply Chain Delay Prediction System

## Overview

This project predicts the likelihood of shipment delays and provides actionable recommendations to improve delivery performance.

## Problem

Delays in supply chain operations affect customer satisfaction and operational efficiency. The goal is to identify high-risk shipments early and support decision-making.

## Solution

* Built a machine learning model to predict delay risk
* Engineered features such as delivery duration and scheduling gaps
* Deployed the model as an interactive Streamlit application
* Added automation to recommend actions based on predicted risk
* Included a what-if simulator to test operational scenarios

## Key Features

* Delay risk prediction (Low, Medium, High)
* Automated recommendations
* Scenario simulation (What-if analysis)
* Downloadable prediction report

## Tech Stack

* Python
* Pandas, Scikit-learn
* Streamlit

## Live App

https://supply-chain-delay-app-nzckm7ysyhpybwd7uqwojj.streamlit.app/

## Project Structure

* `app.py` → Streamlit application
* `notebook/` → Data analysis and modeling
* `requirements.txt` → Dependencies

## Key Insight

Operational factors such as delivery duration and scheduling gaps had more impact on delay risk than external variables in this dataset.

## Future Improvements

* Separate training and inference for production deployment
* Integrate real-time data sources
* Deploy as an API service
