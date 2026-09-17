<p align="center">
  <img src="./assets/logo.svg" alt="FairEnough Logo" width="380" />
</p>

<p align="center">
  <b>Automated AI Bias Auditing & Fairness Mitigation Platform</b>
</p>

<p align="center">
  <a href="https://fairenough-rosy.vercel.app"><img src="https://img.shields.io/badge/Live_Demo-Vercel-000000?style=flat-square&logo=vercel&logoColor=white" alt="Live Demo Vercel"></a>
  <a href="https://fairenough-backend.onrender.com"><img src="https://img.shields.io/badge/API-Render-00E599?style=flat-square&logo=render&logoColor=white" alt="API Render"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-76B900?style=flat-square" alt="License MIT"></a>
</p>

---

## Summary

**FairEnough** is an open-source, enterprise-grade AI fairness auditing and bias mitigation platform. It allows data scientists, AI auditors, compliance teams, and developers to evaluate tabular datasets and ML models for demographic disparities, quantify bias across multiple fairness metrics, diagnose root causes using trained ML models, automatically apply bias mitigation algorithms, and generate plain-language AI explanations alongside professional compliance PDF reports.

---

## What is FairEnough?

FairEnough is an automated end-to-end algorithmic bias diagnosis and remediation tool. When tabular datasets are used for high-stakes automated decisions (such as credit scoring, hiring, medical triage, or loan approvals), inherent historical or collection biases can lead to discriminatory outcomes against protected groups (e.g., based on gender, race, age). 

FairEnough acts as a governance firewall between raw datasets/models and deployment. It calculates standardized fairness metrics, computes an overall **Fairness Score (0–100)** and letter grade (A+ to F), predicts the underlying cause of bias, automatically tests pre-processing and post-processing mitigations (Reweighing, Disparate Impact Remover, Equalized Odds), and leverages **Google Gemini AI** to contextualize findings in accessible natural language.

---

## Why FairEnough is required?

As AI systems become ubiquitous in decision-making, ensuring algorithmic fairness is no longer optional—it is a critical regulatory, ethical, and legal mandate.

1. **Regulatory Compliance**: Regulations such as the **EU AI Act**, **NYC Automated Employment Decisions Law (AEDL Local Law 144)**, and **US FTC algorithmic accountability guidelines** mandate independent audits for high-risk AI models.
2. **Mitigating Systemic Discrimination**: Machine learning models trained on historical data often perpetuate societal inequalities (e.g., lower loan approval rates for protected demographics).
3. **Risk & Reputational Management**: Organizations deploying AI models must prevent costly legal liabilities and public backlash caused by unintended algorithmic bias.
4. **Bridging the AI Governance Gap**: Non-technical stakeholders, auditors, and legal teams require understandable explanations and compliance-ready documentation alongside raw statistical metrics.

---

## Features

- 📊 **Multi-Metric Demographic Fairness Auditing**: Calculates Disparate Impact (DI), Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD).
- 🏷️ **Fairness Scoring & Grading System**: Converts complex multi-dimensional statistical metrics into a single 0–100 score and letter grade (A+ to F).
- 🔍 **ML-Powered Root Cause Classifier**: Utilizes a pre-trained Random Forest classifier to identify whether detected bias stems from *Proxy Features*, *Underrepresentation*, or *Historical Label Skew*.
- 🛠️ **Automated Bias Mitigation Engine**: Evaluates multiple bias correction algorithms (**Reweighing**, **Disparate Impact Remover**, and **Equalized Odds Postprocessing**) and automatically selects the optimal method that restores fairness while preserving model accuracy.
- 🤖 **Google Gemini AI Explanations**: Generates plain-language executive summaries, metric breakdowns, and actionable remediation steps using Google's Gemini models.
- 📄 **Downloadable PDF Audit Reports**: Compiles a sleek, dark-themed, multi-page compliance PDF report formatted with executive summaries, visual charts, metric tables, and audit logs.
- 📁 **Universal Dataset Support**: Accepts CSV, Excel (`.xlsx` / `.xls`), JSON, TSV, Parquet, and ZIP archive uploads up to 100MB.
- 👤 **Account Dashboard & Historic Tracking**: User authentication system (JWT) allowing users to track historic fairness audits, compare improvements over time, and re-download past PDF reports.

---

## How FairEnough works

```
 📥 1. Upload Dataset
       (CSV / Excel / JSON / Parquet)
              │
              ▼
 ⚙️ 2. Select Target & Sensitive Attribute
       (e.g., Target: "approved", Sensitive: "gender")
              │
              ▼
 📊 3. Metric Calculation & ML Root Cause Diagnosis
       (Computes SPD, DI, EOD, AOD & Classifies Cause)
              │
              ▼
 🛠️ 4. Automated Bias Mitigation
       (Evaluates Reweighing, DIR & Equalized Odds)
              │
              ▼
 🤖 5. Gemini AI Executive Narrative Generation
       (Translates math metrics into actionable insights)
              │
              ▼
 📄 6. PDF Report Generation & Dashboard Persistence
       (Download dark-themed audit report or view online)
```

---

## Domain-Agnostic Fairness Auditing

FairEnough is designed to be **fully domain-agnostic**. It is not tied to any specific industry, pre-configured schema, or benchmark dataset (such as Adult Census or COMPAS). Instead, it audits arbitrary tabular CSV datasets across diverse real-world domains — including financial lending, healthcare triage, hiring & recruitment, content moderation, higher education, and insurance.

### Key Principles

- **Industry & Domain Independence**: FairEnough operates on any tabular dataset regardless of domain or application area.
- **Flexible Column Selection**: Works with arbitrary tabular CSV files where users interactively select the **target/outcome column** and one or more **sensitive attributes**.
- **Categorical & Numeric Sensitive Attributes**: Sensitive attributes can be categorical (discrete demographic groups) or numeric (including continuous scales, test scores, or fractional values).
- **No Fixed Schema or Naming Conventions**: Extra feature or metadata columns are freely permitted. There is no required column naming scheme (such as `age`, `gender`, or `income`).
- **Meaningful Outcome Target**: The selected target column must represent a meaningful outcome decision and should generally be binary or categorical (e.g., approved/denied, accepted/rejected, toxic/non-toxic) for demographic fairness analysis.
- **Minimum Group Requirement**: Group-based fairness evaluation mathematically requires at least two distinct demographic groups with sufficient sample representation.
- **Data-Driven Grouping for High Cardinality**: Rather than relying on domain-specific assumptions or hardcoded thresholds, any numeric sensitive attribute with high cardinality (>10 unique values) is automatically partitioned into groups using an empirical **data-driven median split** (`attr <= median` vs `attr > median`).
- **Honest Dataset vs. Model Auditing**:
  - **Dataset-Only Auditing**: When evaluating a standalone dataset without a model, FairEnough computes applicable outcome-based metrics directly from historical decisions — specifically **Statistical Parity Difference (SPD)** and **Disparate Impact (DI)**.
  - **Model-Dependent Auditing**: Error-rate fairness metrics (**Equal Opportunity Difference (EOD)** and **Average Odds Difference (AOD)**) and performance metrics (Accuracy, Precision, Recall, F1) strictly require a compatible trained model with predictions. FairEnough never fabricates model-level metrics when evaluating a dataset in isolation.
- **Transparent Validation & Safeguards**: Datasets with insufficient variation, fewer than two valid groups, or unsuitable target values trigger explicit validation and warning alerts rather than generating misleading or skewed statistics.

### Schema Examples

#### Example 1: Financial Lending & Credit Scoring
Demonstrates a traditional decision pipeline with demographic and economic features:
```csv
age,gender,income,approved
28,Female,62000,1
45,Male,89000,1
52,Female,54000,0
31,Non-Binary,71000,1
```
* **Target Column**: `approved` (Binary decision: `0` = denied, `1` = approved)
* **Sensitive Attribute**: `gender` (Discrete categorical groups) or `age` / `income` (Continuous numeric attributes automatically partitioned by median)
* **Extra Features**: `income`, `age` (Additional feature columns permitted without fixed naming rules)

#### Example 2: Content Moderation & NLP Annotation
Demonstrates an entirely different domain with fractional continuous scores and arbitrary column names:
```csv
post_id,subforum,annotator_score,flagged_as_toxic
p_101,gaming,0.20,0
p_102,news,0.85,1
p_103,sports,0.40,0
p_104,news,0.70,1
```
* **Target Column**: `flagged_as_toxic` (Binary decision: `0` = non-toxic, `1` = toxic)
* **Sensitive Attribute**: `annotator_score` (Continuous fractional agreement ratio partitioned via data-driven median split) or `subforum` (Categorical topic group)
* **Extra Features**: `post_id` (Identifier column ignored by fairness calculations)

---

## Methods Applied

FairEnough combines rigorous statistical fairness formulas with machine learning pre/post-processing algorithms:

### 1. Fairness Metrics

* **Disparate Impact (DI)**: Ratio of positive outcome rates between unprivileged ($D=0$) and privileged ($D=1$) groups.
  $$\text{DI} = \frac{P(\hat{Y}=1 \mid D=0)}{P(\hat{Y}=1 \mid D=1)}$$
  *Ideal range: $0.8 \le \text{DI} \le 1.25$ (Four-Fifths Rule).*

* **Statistical Parity Difference (SPD)**: Difference in positive outcome selection rates.
  $$\text{SPD} = P(\hat{Y}=1 \mid D=0) - P(\hat{Y}=1 \mid D=1)$$
  *Ideal value: $0.0$.*

* **Equal Opportunity Difference (EOD)**: Difference in True Positive Rates (TPR) between groups.
  $$\text{EOD} = \text{TPR}_{D=0} - \text{TPR}_{D=1}$$
  *Ideal value: $0.0$.*

* **Average Odds Difference (AOD)**: Average of True Positive Rate difference and False Positive Rate difference.
  $$\text{AOD} = \frac{1}{2} \left[ (\text{FPR}_{D=0} - \text{FPR}_{D=1}) + (\text{TPR}_{D=0} - \text{TPR}_{D=1}) \right]$$

### 2. Mitigation Methods

* **Reweighing (Pre-Processing)**: Computes instance weights for dataset rows based on group membership and target label to balance outcome rates prior to model training.
* **Disparate Impact Remover (Pre-Processing)**: Edges feature distributions of unprivileged and privileged groups closer together to remove proxy bias while preserving rank ordering.
* **Equalized Odds Postprocessing (Post-Processing)**: Adjusts model decision thresholds to satisfy equalized odds constraints after model predictions are generated.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         REACT FRONTEND                           │
│     (Vite + Tailwind CSS + Framer Motion + Lucide React)        │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ Axios (REST API)
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND                          │
│                                                                  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────────┐  │
│  │ Auth Router     │ │ Analyze Router   │ │ Mitigate Router   │  │
│  │ (JWT Tokens)    │ │ (Fairness Metrics│ │ (AIF360 Engine)   │  │
│  └─────────────────┘ └──────────────────┘ └───────────────────┘  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌───────────────────┐  │
│  │ Explain Router  │ │ Report Router    │ │ Upload Router     │  │
│  │ (Gemini AI)     │ │ (ReportLab PDF)  │ │ (Multi-format)    │  │
│  └─────────────────┘ └──────────────────┘ └───────────────────┘  │
└──────┬──────────────────────┬──────────────────────┬─────────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────────┐
│ MONGODB      │      │ GOOGLE       │      │ SCIPY / AIF360 / │
│ ATLAS        │      │ GEMINI API   │      │ SCIKIT-LEARN     │
│ (User & Audit│      │ (Executive   │      │ (Fairness & ML   │
│ Persistence) │      │ Narrative)   │      │ Algorithms)      │
└──────────────┘      └──────────────┘      └──────────────────┘
```

---

## Tech stack

### Frontend

| Component | Library / Framework | Official Documentation Link |
|---|---|---|
| Framework | React.js (v18+) | [https://react.dev](https://react.dev) |
| Build Tool | Vite | [https://vitejs.dev](https://vitejs.dev) |
| Styling | Tailwind CSS | [https://tailwindcss.com](https://tailwindcss.com) |
| Animations | Framer Motion | [https://www.framer.com/motion](https://www.framer.com/motion) |
| Icons | Lucide React | [https://lucide.dev](https://lucide.dev) |
| HTTP Client | Axios | [https://axios-http.com](https://axios-http.com) |

### Backend

| Component | Library / Framework | Official Documentation Link |
|---|---|---|
| Framework | FastAPI | [https://fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Web Server | Uvicorn | [https://www.uvicorn.org](https://www.uvicorn.org) |
| Language | Python (v3.12) | [https://www.python.org](https://www.python.org) |
| AI / LLM | Google GenAI SDK | [https://ai.google.dev](https://ai.google.dev) |
| ML & Fairness | AIF360 / Scikit-Learn / Pandas | [https://aif360.mybluemix.net](https://aif360.mybluemix.net) |
| Database Driver | Motor (Async MongoDB) | [https://motor.readthedocs.io](https://motor.readthedocs.io) |
| PDF Engine | ReportLab | [https://www.reportlab.com](https://www.reportlab.com) |

---

## API

The backend provides OpenAPI / Swagger documentation at `http://localhost:8000/docs`.

### Key Endpoints

#### 1. Authentication
* `POST /api/auth/register` — Register a new account.
* `POST /api/auth/login` — Authenticate and receive a JWT token.

#### 2. Dataset Processing & Analysis
* `POST /api/upload/` — Upload a tabular dataset (`.csv`, `.xlsx`, `.parquet`, `.json`, `.zip`).
* `POST /api/analyze/` — Compute fairness metrics, overall score, letter grade, and root cause diagnosis.
* `POST /api/mitigate/` — Run automated bias mitigation algorithms and return corrected metrics.

#### 3. AI & Reporting
* `POST /api/explain/` — Fetch Google Gemini narrative explanation for audit metrics.
* `POST /api/report/generate-pdf` — Compile and return a downloadable PDF audit report.
* `GET /api/reports/my-reports` — Retrieve user historic audit records.

---

## Example

### API Request: Analyze Dataset Fairness

```json
POST /api/analyze/
Content-Type: application/json

{
  "dataset_id": "ds_9876543210",
  "target_column": "loan_approved",
  "sensitive_column": "gender",
  "privileged_group": 1,
  "unprivileged_group": 0
}
```

### API Response

```json
{
  "status": "success",
  "fairness_score": 68.5,
  "letter_grade": "C",
  "metrics": {
    "disparate_impact": 0.64,
    "statistical_parity_difference": -0.22,
    "equal_opportunity_difference": -0.18,
    "average_odds_difference": -0.15
  },
  "bias_detected": true,
  "root_cause": "Historical Label Skew & Proxy Feature Imbalance",
  "recommended_mitigation": "Reweighing"
}
```

---

## General Installation & Setup

### Prerequisites

Ensure you have the following installed on your local system:
- **Node.js**: `v18.0.0` or higher
- **npm**: `v9.0.0` or higher
- **Python**: `v3.10` or higher
- **MongoDB**: Local MongoDB instance or MongoDB Atlas URI
- **Google Gemini API Key**: Obtainable from [Google AI Studio](https://aistudio.google.com/)

---

### 1. Repository Cloning

```bash
git clone https://github.com/tanmayjhanjhari/fairenough.git
cd fairenough
```

---

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Create environment config
cp .env.example .env
```

Edit `.env` to configure your credentials:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
MONGODB_URI=mongodb://localhost:27017/fairenough
SECRET_KEY=your_super_secret_jwt_key
```

Start backend development server:
```bash
uvicorn main:app --reload --port 8000
```

---

### 3. Frontend Setup

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment config (optional)
cp .env.example .env
```

Start frontend development server:
```bash
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Deployment

### Live Application Links

| Component | Platform | Deployment Status / Link |
|---|---|---|
| **Frontend Web App** | Vercel | [https://fairenough-demo.vercel.app](https://fairenough-demo.vercel.app) *(Space for live Vercel link)* |
| **Backend REST API** | Render / Railway | [https://fairenough-api.onrender.com](https://fairenough-api.onrender.com) *(Space for live Render/Railway link)* |

### Deployment Steps

1. **Frontend (Vercel)**:
   - Connect your GitHub repository to Vercel.
   - Set build command: `npm run build` and output directory: `dist`.
   - Set environment variable `VITE_API_BASE_URL` pointing to your deployed backend URL.

2. **Backend (Render / Railway / Docker)**:
   - Deploy using the provided `Dockerfile` or Render Web Service configuration.
   - Set environment variables (`GEMINI_API_KEY`, `MONGODB_URI`, `SECRET_KEY`).

---

## Performance

- ⚡ **Sub-second Fairness Computations**: Optimized vector math via NumPy/Pandas processes datasets up to 100,000 rows in under 800ms.
- 🚀 **Asynchronous REST Architecture**: Powered by FastAPI async request handling and Motor non-blocking database queries.
- ⚡ **Stream-Ready PDF Generation**: PDF report generation via ReportLab compiles full multi-page visual reports in under 1.5 seconds.
- 🏎️ **Optimized Frontend Bundle**: Built with Vite and code-splitting, ensuring sub-1 second initial page loads and smooth 60fps animations.

---

## Future Scalability

- 📦 **Distributed Task Workers**: Integration with Celery & Redis for asynchronous background processing of multi-gigabyte datasets.
- 🧠 **LLM & RAG Bias Auditing**: Expanding beyond tabular data to evaluate text embeddings, prompt outputs, and RAG pipelines for bias.
- 🔌 **CI/CD Pipeline Plugins**: GitHub Actions & GitLab CI integrations to automatically block deployments if model bias metrics exceed allowed thresholds.
- 🌐 **Multi-tenant Enterprise Workspaces**: Organization-level RBAC (Role-Based Access Control) with team audit logs and policy enforcement.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
