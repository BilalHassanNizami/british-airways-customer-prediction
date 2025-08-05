![Project Banner](cover.png)


# Predicting Customer Purchase Behavior – British Airways (Random Forest Model)

This project builds and evaluates a Random Forest classifier to predict customer purchase behavior based on travel data from British Airways. The goal is to identify high-potential customers and optimize marketing strategies.

## Project Highlights

- Cleaned and explored flight booking data
- Built and evaluated a Random Forest model
- Identified top 15 predictive features
- Saved the trained model using `joblib`
- Deployed the model using Python and TabPy for Tableau integration

## Folder Structure

- `data/`: cleaned dataset
- `model/`: trained Random Forest model
- `notebooks/`: Jupyter notebook with full pipeline
- `assets/`: Tableau dashboard or visual examples
- `README.md`: this file
- `requirements.txt`: Python libraries

## Tech Stack

- Python (Pandas, Scikit-learn, imbalanced-learn, Matplotlib)
- Jupyter Notebook
- Joblib for model saving
- TabPy + Tableau for visualization (Optional)

## Top 5 Features Identified

1. `flight_hour`
2. `purchase_lead`
3. `length_of_stay`
4. `num_passengers`
5. `route_importance_score`

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/RandomForest_BritishAirways.git
cd RandomForest_BritishAirways

# Install dependencies
pip install -r requirements.txt

