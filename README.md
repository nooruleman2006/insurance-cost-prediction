<div align="center">

# 🏥 Medical Insurance Cost Predictor

### Predict a person's yearly medical insurance charge with machine learning

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-FFD21E?style=for-the-badge)](https://nooruleman2006oct-insurance-cost-predictor.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-FFCC00?style=for-the-badge)](https://catboost.ai/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316?style=for-the-badge)](https://www.gradio.app/)
[![Colab](https://img.shields.io/badge/Notebook-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**[👉 Try the live app](https://nooruleman2006oct-insurance-cost-predictor.hf.space/)**

<img width="1653" height="927" alt="image" src="https://github.com/user-attachments/assets/5be712c7-b580-488f-a4c8-3e93781ae2fa" />

</div>

---

## 📌 Overview

This project is an end-to-end **regression** pipeline. It predicts the yearly **medical insurance charge (in US dollars)** of a person from six simple details: age, sex, BMI, number of children, smoker status and region.

Five modern regression algorithms were compared with **10-fold cross validation**, and the best one (**CatBoost**) was trained on the full dataset, saved, and deployed as a public web app.

| | |
|---|---|
| **Task** | Regression |
| **Target** | `charges` (yearly insurance cost, $) |
| **Evaluation measure** | RMSE (Root Mean Squared Error, in $) |
| **Validation** | 10-fold cross validation (about 90% train / 10% test in every round) |
| **Best model** | CatBoost, mean RMSE of **$4,731.81** |
| **Deployment** | Gradio app on Hugging Face Spaces |

---

## 🎬 Demo Video

> 📹 **Watch the 1-minute walkthrough:**

<!-- Replace this line with your video. See the "How to add the video" note at the bottom of this file. -->




---

## 📸 Screenshots

<div align="center">


<img width="1512" height="912" alt="image" src="https://github.com/user-attachments/assets/e0e19968-ba5e-4fc6-b410-06387d5e4745" />
<img width="1510" height="926" alt="image" src="https://github.com/user-attachments/assets/5f1cbe67-28b1-4a17-888b-420c6dd6a209" />
<img width="1528" height="689" alt="image" src="https://github.com/user-attachments/assets/b217a181-af35-45d3-827d-856b548ed8e7" />


</div>

---

## 📊 Dataset

**Medical Cost Personal Dataset** (`insurance.csv`)

- **1,338 records**, 7 columns (1,337 after removing 1 duplicate row)
- No missing values

| Feature | Description | Type |
|---|---|---|
| `age` | Age of the person (18–64) | Numerical |
| `sex` | male / female | Categorical |
| `bmi` | Body mass index | Numerical |
| `children` | Number of children covered (0–5) | Numerical |
| `smoker` | yes / no | Categorical |
| `region` | northeast / northwest / southeast / southwest | Categorical |
| **`charges`** | **Yearly medical insurance cost ($), the value we predict** | **Numerical (target)** |

Charges are **right-skewed**: the mean is about $13,270, the median about $9,382, and the maximum $63,770. A few very expensive cases raise the RMSE, which is worth keeping in mind when reading the results.

---

## 🔬 Methodology

1. **Load** the full dataset.
2. **Understand** it: check types, missing values, duplicates and the spread of the target.
3. **Pre-process**: remove the duplicate row and separate the features (X) from the target (y).
4. **Pipeline**: numeric columns are median-filled and standard-scaled, categorical columns are one-hot encoded. This is done *inside* the cross validation loop, so the test fold never leaks into training.
5. **Compare 5 algorithms** with the same 10 folds (`shuffle=True`, `random_state=42`), all with default settings for a fair comparison.
6. **Select** the algorithm with the lowest mean RMSE.
7. **Train** it on all 1,337 records and **save** the full pipeline with `pickle`.
8. **Deploy** it as a Gradio web app on Hugging Face Spaces.

### Algorithms compared

| # | Algorithm | Library |
|---|---|---|
| 1 | XGBoost | `xgboost` |
| 2 | LightGBM | `lightgbm` |
| 3 | CatBoost | `catboost` |
| 4 | HistGradientBoosting | `scikit-learn` |
| 5 | Random Forest | `scikit-learn` |

---

## 🏆 Results

Mean RMSE over 10 folds (**lower is better**):

| Rank | Algorithm | Mean RMSE ($) | Std Dev ($) |
|:---:|---|---:|---:|
| 🥇 1 | **CatBoost** | **4,731.81** | 438.32 |
| 🥈 2 | LightGBM | 4,754.27 | 421.24 |
| 🥉 3 | HistGradientBoosting | 4,765.95 | 401.93 |
| 4 | Random Forest | 4,824.22 | 391.41 |
| 5 | XGBoost | 5,160.96 | 479.26 |

**Key takeaways**

- **CatBoost** achieved the lowest mean RMSE and was selected as the final model.
- The top four algorithms are within about $92 of each other, which is smaller than the fold-to-fold variation (about $400), so any of them would be a reasonable choice.
- **XGBoost** was the weakest with default settings and had the worst single fold ($6,171.91).
- An RMSE of about $4,700 is roughly 36% of the average charge. This is typical for this dataset, where high-BMI smokers have very high charges.

---

## 🚀 Live Deployment

The trained model is deployed on **Hugging Face Spaces**:

**🔗 https://nooruleman2006oct-insurance-cost-predictor.hf.space/**

The app has:

- Sliders and selectors for the six inputs
- A large result card with the **yearly** and **monthly** estimate and the typical model error
- **Light mode by default**, with a **Dark mode** button in the top-right corner
- A responsive layout that uses 70% of the screen width on desktop

> ⏳ Free Hugging Face Spaces go to sleep when idle. If the page shows "Sleeping" or "Building", wait a minute for it to wake up.

---

## 🗂️ Project Structure

```
insurance-cost-prediction/
├── insurance_cost_prediction_10fold.ipynb   # Full notebook: data, 10-fold CV, RMSE comparison, model, app
├── insurance.csv                            # Dataset
├── insurance_model.pkl                      # Trained CatBoost pipeline
├── app.py                                   # Gradio web app
├── requirements.txt                         # Pinned library versions
├── screenshots/                             # Images used in this README
└── README.md
```

---

## 💻 Run It Yourself

### Option 1: Google Colab (recommended)

1. Open `insurance_cost_prediction_10fold.ipynb` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**).
2. Click **Runtime → Run all**.
3. The notebook downloads the dataset, runs the 10-fold comparison, trains and saves the model, and launches the app.

### Option 2: Run the app locally

```bash
git clone https://github.com/YOUR-USERNAME/insurance-cost-prediction.git
cd insurance-cost-prediction

pip install -r requirements.txt
pip install gradio

python app.py
```

Then open the local URL printed in the terminal (usually `http://127.0.0.1:7860`).

> The saved model was trained with the library versions listed in `requirements.txt`. Use the same versions to load `insurance_model.pkl` without warnings.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data and ML:** pandas, NumPy, scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualisation:** Matplotlib
- **Web app:** Gradio
- **Hosting:** Hugging Face Spaces
- **Environment:** Google Colab

---

## ⚠️ Limitations

- The model uses **default hyper-parameters** (no tuning), to keep the comparison simple and fair.
- The dataset is small (1,337 records), so predictions for rare combinations of inputs are less reliable.
- Predictions are **estimates for educational purposes**, not real insurance quotes.

## 🔮 Future Work

- Tune hyper-parameters (for example with `GridSearchCV` or Optuna)
- Try predicting the logarithm of the charges to reduce the effect of very expensive cases
- Add feature-importance and explanation charts (for example SHAP) to the app

---

## 👤 Author

**YOUR NAME**
🔗 [GitHub][https://github.com/nooruleman2006] · [LinkedIn](https://www.linkedin.com/in/nooruleman2006)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star.

</div>

<!--
HOW TO ADD THE VIDEO (delete this comment when done)
1. Click the pencil icon on README.md in GitHub to edit it.
2. Drag your demo.mp4 (under 10 MB) into the editor. GitHub uploads it and inserts a link.
3. Replace the line "**[ADD DEMO VIDEO HERE]**" with that link on its own line.
   Alternative: upload the video to YouTube and use a clickable thumbnail:
   [![Demo video](screenshots/01-light-mode.png)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)
-->
