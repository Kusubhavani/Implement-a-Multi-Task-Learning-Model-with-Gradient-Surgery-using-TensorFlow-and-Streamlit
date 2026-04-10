#  Multi-Task Learning with Gradient Surgery (PCGrad)

##  Project Overview

This project implements a **Multi-Task Learning (MTL)** system using TensorFlow, enhanced with **Gradient Surgery (PCGrad)** to resolve gradient conflicts between tasks.

The system trains a shared neural network to perform **two tasks simultaneously**:

* Task A: Binary Classification
* Task B: Regression

A **custom training loop** is used to:

* Detect gradient conflicts using cosine similarity
* Apply PCGrad to resolve conflicts
* Improve overall learning performance

A **Streamlit dashboard** is built to visualize:

* Gradient conflicts over time
* Model performance comparison

---

##  Key Concepts

### 🔹 Multi-Task Learning (MTL)

A single model learns multiple tasks using a shared representation.

###  Gradient Conflict

Occurs when gradients from different tasks point in opposite directions.

###  PCGrad (Projected Conflicting Gradients)

Resolves gradient conflicts by projecting gradients to avoid interference.

---
## Demo Link:
https://drive.google.com/file/d/1X3uZ3J9PNLKWPijM_qIwAhFT62BCcVCy/view?usp=drive_link
##  Project Structure

```
mtl-pcgrad-project/
│
├── app/
│   └── streamlit_app.py        # Streamlit dashboard
│
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── __init__.py
│
├── models/
│   ├── model.py                # Multi-task model
│   └── __init__.py
│
├── training/
│   ├── train_baseline.py       # Baseline training (naive loss)
│   ├── train_pcgrad.py         # PCGrad training
│   ├── generate_final_metrics.py
│   └── __init__.py
│
├── results/                    # Output files (auto-generated)
│
├── generate_model_summary.py   # Model architecture export
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

##  Installation

### 1. Clone the repository

```
git clone https://github.com/Kusubhavani/Implement-a-Multi-Task-Learning-Model-with-Gradient-Surgery-using-TensorFlow-and-Streamlit.git
cd Implement-a-Multi-Task-Learning-Model-with-Gradient-Surgery-using-TensorFlow-and-Streamlit
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

##  Execution Steps

###  Step 1: Train Baseline Model

```
python -m training.train_baseline
```

Output:

```
results/baseline_metrics.csv
```

---

###  Step 2: Train PCGrad Model

```
python -m training.train_pcgrad
```

Outputs:

```
results/pcgrad_metrics.csv
results/gradient_conflict.csv
```

---

### Step 3: Generate Model Summary

```
python generate_model_summary.py
```

Output:

```
results/model_architecture.txt
```

---

### Step 4: Generate Final Metrics JSON

```
python -m training.generate_final_metrics
```

Output:

```
results/final_metrics.json
```

---

###  Step 5: Run Streamlit Dashboard

```
streamlit run app/streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

##  Run with Docker (Recommended)

### Build and run the project:

```
docker-compose up --build
```

Access dashboard:

```
http://localhost:8501
```

---

##  Dashboard Features

### 🔹 Gradient Conflict Monitor

* Displays cosine similarity between task gradients
* 🔴 Highlights negative values (conflicts)

### 🔹 Model Performance

* Compares baseline vs PCGrad
* Tracks loss for both tasks

---

## Output Files

| File                     | Description                |
| ------------------------ | -------------------------- |
| `baseline_metrics.csv`   | Baseline model performance |
| `pcgrad_metrics.csv`     | PCGrad model performance   |
| `gradient_conflict.csv`  | Cosine similarity logs     |
| `model_architecture.txt` | Model summary              |
| `final_metrics.json`     | Final comparison metrics   |
| `analysis.md`            | Project analysis           |

---

## Results Summary

* Gradient conflicts are clearly observed during training
* PCGrad successfully reduces interference
* Model shows stable convergence across both tasks

---

## Future Improvements

* Add UMAP visualization for feature space
* Include accuracy and F1-score metrics
* Use real-world datasets (CelebA, NLP datasets)
* Enhance dashboard UI with tabs and filters

---

## Author

Bhavani Kusu
B.Tech – Artificial Intelligence & Machine Learning

---

## 📜 License

This project is for educational purposes.
