# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

## 📌 Overview

This project performs **Exploratory Data Analysis (EDA)** on the Titanic dataset using **Python, Seaborn, and Plotly**.
The goal is to understand patterns, relationships, and key factors affecting passenger survival.

---

## 📂 Dataset

* Dataset: **Titanic Dataset (Seaborn)**
* Contains information about passengers such as:

  * Age
  * Gender
  * Passenger Class
  * Fare
  * Survival Status

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Seaborn (for static visualization)
* Matplotlib
* Plotly (for interactive visualization)

---

## 🧹 Data Cleaning

The dataset contains missing values, which were handled as follows:

* Filled missing **Age** values using median
* Filled missing **Embarked** values using mode
* Handled **Deck** categorical missing values by adding a new category `"Unknown"`
* Removed unnecessary columns

---

## 📊 Visualizations

### 🔹 Static Visualizations (Seaborn)

* Bar Plot → Survival count by gender
* Histogram → Age distribution
* Pie Chart → Survival ratio
* Box Plot → Age vs Passenger Class

### 🔹 Interactive Visualizations (Plotly)

* Interactive Bar Chart
* Interactive Histogram
* Interactive Pie Chart
* Interactive Box Plot

---

## 📈 Key Insights

* Females had a higher survival rate than males
* Passengers in **1st class** had better survival chances
* Fare and passenger class are strongly related
* Age alone is not a strong predictor of survival

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
```

2. Install required libraries:

```bash
pip install pandas numpy seaborn matplotlib plotly
```

3. Run the Python file or Jupyter Notebook

---

## 📁 Project Structure

```
├── eda_titanic.ipynb / eda_titanic.py
├── README.md
```

---

## 🎯 Purpose

This project helps in:

* Understanding EDA concepts
* Practicing data visualization
* Learning data cleaning techniques
* Building a strong foundation for data science

---

## 🤝 Contributing

Feel free to fork this repository and improve it.

---

## ⭐ Support

If you found this useful, give it a ⭐ on GitHub!
