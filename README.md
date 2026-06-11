# ⚡ Smart Electricity Bill Predictor

> An AI-powered desktop application that predicts monthly electricity bills, explains cost drivers, and provides intelligent saving recommendations — built with Python, Random Forest, and SHAP.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-006400?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

---

## 📌 Overview

**Smart Electricity Bill Predictor** is a four-tab desktop application that combines machine learning with a clean dark-themed GUI to help households understand and reduce their electricity costs.

| Tab | Purpose |
|-----|---------|
| 🏠 **Predictor** | Enter appliance counts + usage hours → get instant bill estimate |
| 📊 **Bill Breakdown** | SHAP-powered visualization showing exactly what's driving your bill |
| 💡 **Insights** | Savings analysis with appliance-level cost contributions |
| 🤖 **AI Assistant** | ChatGPT-style chat interface for personalized energy advice |

---

## ✨ Features

- **ML-powered predictions** using a Random Forest Regressor (R² = 0.98)
- **SHAP explainability** — understand *why* your bill is what it is
- **Appliance-level cost breakdown** — see how much each device contributes in PKR
- **Seasonal adjustment** — model accounts for higher summer consumption (May–Sep +15%)
- **AI chat assistant** powered by Free ChatGPT API with conversation context
- **Dark theme UI** built entirely with Tkinter (no external GUI framework needed)
- **Fully offline** — predictions run locally, no internet required (except AI tab)

---

## 🖼️ Screenshots

> Run `python app.py` to see the full UI.

**Tab 1 — Bill Predictor**
- Input fields for: Fan, Refrigerator, AC, TV, Monitor, Motor Pump, Month, Daily Hours, Tariff Rate
- One-click prediction with animated result display

**Tab 2 — Bill Breakdown (SHAP)**
- Horizontal bar chart showing each appliance's PKR contribution
- Color-coded by impact level

**Tab 3 — Insights & Savings**
- Real-time savings analysis
- Appliance usage optimization suggestions

**Tab 4 — AI Assistant**
- ChatGPT-style bubble interface
- Context-aware responses using your current appliance values

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/abdul-wahid-lab/smart-electricity-bill-predictor.git
cd smart-electricity-bill-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python generate_data.py

# 4. Train the model
python retrain.py

# 5. Launch the app
python app.py
```

### AI Assistant Setup (Optional)
The AI Assistant tab requires a free RapidAPI key:

1. Sign up at [RapidAPI](https://rapidapi.com)
2. Subscribe to the **Free ChatGPT API**
3. Create a `.env` file in the project root:

```env
RAPIDAPI_KEY=your_api_key_here
```

---

## 📁 Project Structure

```
smart-electricity-bill-predictor/
│
├── app.py                      # Main application (4-tab Tkinter GUI)
├── generate_data.py            # Dataset generator (50,000 synthetic records)
├── retrain.py                  # Model training script
├── train_model.ipynb           # Jupyter notebook (exploratory training)
│
├── electricity_bill_dataset.csv  # Generated dataset
├── features.pkl                  # Saved feature list
├── electricity_model.pkl         # Trained model (generated locally — not in repo)
│
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed — see .gitignore)
└── .gitignore
```

---

## 🧠 How the Model Works

### Dataset
- **50,000 synthetic records** generated with realistic appliance wattages
- Appliances: Fan (75W), Refrigerator (150W), AC (1500W), TV (100W), Monitor (50W), Motor Pump (750W)
- Formula: `Bill = (total_kW × DailyHours × 30 days × TariffRate) × seasonal_factor`
- 5% random noise added for realism

### Features
| Feature | Description | Range |
|---------|-------------|-------|
| Fan | Number of fans | 0–10 |
| Refrigerator | Number of fridges | 0–3 |
| AirConditioner | Number of ACs | 0–3 |
| Television | Number of TVs | 0–4 |
| Monitor | Number of monitors | 0–3 |
| MotorPump | Number of pumps | 0–2 |
| Month | Month of year | 1–12 |
| DailyHours | Avg hours/day all appliances run | 4–18 |
| TariffRate | Electricity rate (PKR/kWh) | 5–25 |

### Model Performance
- **Algorithm:** Random Forest Regressor (100 trees)
- **R² Score:** 0.9819
- **Training split:** 80/20

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| GUI | Python Tkinter |
| ML Model | scikit-learn (Random Forest) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Matplotlib |
| Data | Pandas, NumPy |
| AI Chat | RapidAPI Free ChatGPT |
| Config | python-dotenv |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
joblib
shap
matplotlib
requests
python-dotenv
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👨‍💻 Authors

- **Abdul Wahid** — [abdul-wahid-lab](https://github.com/abdul-wahid-lab)
- **Hamza Akhlaque** — [hamzaakhlaq2002](https://github.com/hamzaakhlaq2002)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for the Random Forest implementation
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [RapidAPI](https://rapidapi.com/) for the free ChatGPT API endpoint
