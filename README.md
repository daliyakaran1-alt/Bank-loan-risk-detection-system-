# FinSight: Bank Risk Analytics Platform

FinSight is a professional, enterprise-grade banking risk analytics platform designed to evaluate loan applicant default probability using a lightweight interactive Machine Learning engine. It provides real-time data integration, explainable AI (XAI) for decision logic, and comprehensive dashboard reporting.

## 🚀 Key Features

*   **Real-Time Executive Dashboard**: High-level KPIs, default rates, and financial exposure updated instantly as new applications are processed.
*   **Predictive ML Engine**: Evaluates applicant risk (Low/Medium/High) using a custom Logistic Regression model trained on over 300,000 historical banking records.
*   **Explainable AI (XAI)**: Generates human-readable explanations detailing *why* an applicant received a specific risk score (e.g., "Loan-to-Income ratio is elevated").
*   **Dynamic Application History**: A searchable, filterable ledger of all processed applications with one-click detailed report modals.
*   **Comparative Analytics**: Run two applicants side-by-side to evaluate comparative risk exposure.
*   **Data Analytics**: Visual charts mapping out income distributions, age brackets, and credit scores against historical default rates.

## 🛠️ Technology Stack

### Backend & Architecture
*   **Framework**: Flask (Python)
*   **Database**: SQLite with SQLAlchemy ORM
*   **Machine Learning**: Scikit-learn (Logistic Regression, Label Encoding)
*   **Data Processing**: Pandas, NumPy
*   **Authentication**: Bcrypt password hashing & Flask Session management

### Frontend & UI/UX
*   **Templating**: Jinja2
*   **Styling**: Tailwind CSS (via CDN)
*   **Typography**: Google Fonts (Inter)
*   **Icons**: Heroicons (SVG)
*   **Animations**: Animate.css & Custom Tailwind Keyframes
*   **Charts**: Chart.js

## 🧠 Machine Learning Model

The core prediction engine is built on a cleaned subset of banking data (`application_train.csv`).
*   **Algorithm**: Logistic Regression (`class_weight='balanced'`)
*   **Features Used**: 12 critical data points (Income, Age, Loan Amount, Credit Score, Employment Years, Children, Gender, Education, Family Status, Car Ownership, Real Estate Ownership, Housing Type).
*   **Serialization**: Models and encoders are saved via `joblib` for high-speed inference without retraining.

## ⚙️ Local Setup & Installation

1.  **Clone the Repository**
2.  **Create a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    Make sure you have Flask, Pandas, Scikit-learn, SQLAlchemy, and bcrypt installed.
    ```bash
    pip install flask pandas numpy scikit-learn sqlalchemy bcrypt joblib
    ```
4.  **Run the Application**:
    ```bash
    python flask_app.py
    ```
5.  **Access the Platform**:
    Open your browser and navigate to `http://localhost:5000`

### Default Credentials
*   **Admin**: `admin` / `admin`
*   **Executive**: `exec` / `exec`
*   **Loan Officer**: `loanofficer` / `loanofficer`
*   **Risk Analyst**: `risk` / `risk`

## 📁 Project Structure

*   `/flask_app.py`: Main application routing and API endpoints.
*   `/src/db/`: Database configuration, ORM models, and CRUD operations.
*   `/src/services/`: Core business logic (`InteractiveRiskService`, `DataService`, `AuthService`).
*   `/scripts/`: Offline ML training scripts.
*   `/templates/`: Jinja2 HTML templates for the frontend.
*   `/models/trained/`: Serialized `.joblib` machine learning models.
*   `/data/`: Raw and processed dataset storage.
