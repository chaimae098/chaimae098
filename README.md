<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   CHAIMAE KAZOURY  ·  AI/CS Engineer  ·  ENSAM Casablanca   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&duration=3000&pause=800&color=C8522A&center=true&vCenter=true&multiline=true&width=600&height=60&lines=Anomaly+Detection+%7C+Real-Time+Systems+%7C+ML+Pipelines;Sensor+data+in+%E2%86%92+Predictions+out+%E2%86%92+Failures+caught+early)](https://git.io/typing-svg)

<a href="https://linkedin.com/in/chaimae-kazoury-040715238"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white"/></a>
<a href="mailto:chaimaekaz05@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white"/></a>
<a href="https://chaimae098.github.io"><img src="https://img.shields.io/badge/Portfolio-000000?style=flat-square&logo=github&logoColor=white"/></a>
<img src="https://komarev.com/ghpvc/?username=chaimae098&style=flat-square&color=c8522a&label=profile+views"/>

</div>

---

## `$ whoami`

3rd-year **Computer Science & AI Engineering** student at **ENSAM Casablanca**.  
I build systems that catch what humans miss — anomalies in sensor streams, failures before they cascade, patterns buried in time-series noise.

Currently engineering an end-to-end **industrial ML monitoring system**: raw sensor data → preprocessing pipeline → ML classifier → live React dashboard with real-time failure alerts.

```python
chaimae = {
    "focus"      : ["anomaly detection", "real-time monitoring", "ML pipelines"],
    "currently"  : "Machine Failure Detection — industrial IoT end-to-end system",
    "exploring"  : ["time-series forecasting", "geospatial data systems"],
    "certified"  : "Oracle Certified Professional — Java SE 17 (2026)",
    "languages"  : ["🇫🇷 French C1", "🇬🇧 English C1", "🇲🇦 Arabic (native)"],
}
```

---

## `$ ls projects/`

<details>
<summary><b>🔴 Machine Failure Detection</b> — <i>Industrial IoT · Anomaly Detection · Full-Stack ML</i></summary>

<br>

**Problem solved:** Machines fail without warning — costly downtime, lost production. This system catches the signal *before* the breakdown.

**My role:** Designed the full system end-to-end — from raw sensor ingestion to live dashboard.

**What I built:**
- Full preprocessing pipeline: normalization, missing-value imputation, feature engineering on sensor signals
- Supervised ML classifier (Scikit-learn) to classify machine health state
- Django REST API exposing model predictions
- React dashboard with real-time alert system

**Stack:** `Python` `Scikit-learn` `Django` `React` `PostgreSQL`

```python
# Feature engineering on sensor signals
def compute_rolling_features(df, window=10):
    df['mean_vibration'] = df['vibration'].rolling(window).mean()
    df['std_vibration']  = df['vibration'].rolling(window).std()
    df['peak_to_peak']   = df['vibration'].rolling(window).max() \
                         - df['vibration'].rolling(window).min()
    return df.dropna()
```

🔗 [**github.com/chaimae098/Machine\_failure\_detection**](https://github.com/chaimae098/Machine_failure_detection) · `in progress`

</details>

---

<details>
<summary><b>🟢 prmon Anomaly Detection</b> — <i>Process Monitoring · Statistical Methods · Time-Series</i></summary>

<br>

**Problem solved:** Process monitoring metrics (prmon) accumulate silently — nobody notices drift until it's too late.

**My role:** Designed the detection pipeline, chose and combined statistical methods, made it generic enough to plug into any sensor stream.

**What I built:**
- Time-series analysis on system monitoring data
- Combined z-score, IQR, and rolling statistics for robust flagging
- Reusable pipeline — drop any sensor CSV in, anomalies come out

**Stack:** `Python` `Pandas` `NumPy` `Time-Series Analysis`

```python
def detect_anomalies(series, window=20, z_thresh=3.0):
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    z_scores     = (series - rolling_mean) / rolling_std

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr     = q3 - q1

    anomalies = (z_scores.abs() > z_thresh) | \
                (series < q1 - 1.5*iqr) | \
                (series > q3 + 1.5*iqr)
    return anomalies
```

🔗 [**github.com/chaimae098/prmon-anomaly-detection**](https://github.com/chaimae098/prmon-anomaly-detection)

</details>

---

<details>
<summary><b>🔵 IncidAI — Intelligent Incident Classification</b> — <i>NLP · Microservices · Full-Stack AI</i></summary>

<br>

**Problem solved:** IT teams waste hours manually triaging incident tickets. IncidAI automates classification and surfaces similar past resolutions instantly.

**My role:** Led the ML pipeline (XGBoost + BERT embeddings) and the semantic search component (FAISS). Integrated everything into the FastAPI microservice.

**What I built:**
- XGBoost classifier on BERT semantic embeddings for ticket classification
- FAISS vector index for fast semantic search over resolved incidents
- FastAPI ML microservice consumed by a Django REST backend
- React UI for real-time classification and resolution suggestions

**Stack:** `XGBoost` `BERT` `FAISS` `FastAPI` `Django` `React`

```python
# Semantic search over resolved incidents
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_index(resolved_tickets):
    embeddings = model.encode(resolved_tickets)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    return index

def find_similar(query, index, resolved_tickets, k=3):
    q_emb = model.encode([query]).astype('float32')
    _, ids = index.search(q_emb, k)
    return [resolved_tickets[i] for i in ids[0]]
```

🔗 [**github.com/AgorAI-Hackathon/IncidAI**](https://github.com/AgorAI-Hackathon/IncidAI)

</details>

---

## `$ cat skills.txt`

```
┌─────────────────────────────────────────────────────────┐
│  AI & Anomaly Detection                                 │
│  ── Isolation Forest, z-score, IQR, rolling stats      │
│  ── XGBoost, BERT embeddings, supervised classification │
│  ── Model evaluation, false positive reduction          │
│                                                         │
│  Data Engineering                                       │
│  ── Sensor data preprocessing & feature engineering    │
│  ── Time-series analysis · Pandas · NumPy               │
│                                                         │
│  Backend & APIs                                         │
│  ── Django (intermediate) · FastAPI · REST APIs         │
│  ── Spring Boot · Laravel (intermediate)                │
│                                                         │
│  Frontend & Dashboards                                  │
│  ── React · JavaScript · HTML/CSS                       │
│                                                         │
│  Databases                                              │
│  ── PostgreSQL · Oracle SQL · Relational design         │
│                                                         │
│  Languages                                              │
│  ── Python (advanced) · Java (Oracle certified)         │
│  ── C · JavaScript · C++ (beginner)                     │
│                                                         │
│  Tools                                                  │
│  ── Git · Linux (Ubuntu) · Postman · Scrum              │
└─────────────────────────────────────────────────────────┘
```

---

## `$ cat experience.log`

```
[Jan–Mar 2026]  Web Dev Intern · S B Solutions, Casablanca
                → Full-stack platform: React + Spring Boot
                → Connected clients with local service providers
                → Code reviews · Sprint planning · Feature delivery

[Jun–Aug 2025]  IT Intern · Fiscinfo, Casablanca
                → Internal data analysis & validation
                → Cross-team workflows for data reporting pipelines
```

---

## `$ cat certifications.txt`

```
[2026]  Oracle Certified Professional — Java SE 17 Developer  🏅
[2025]  Oracle Cloud Infrastructure Foundations Associate     ☁️
        Data Visualization — Kaggle                           📊
        C Essentials 1 — Cisco                                💻
```

---

## `$ git log --stats`

<div align="center">

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=chaimae098&show_icons=true&theme=default&hide_border=true&title_color=c8522a&icon_color=c8522a&text_color=333333&bg_color=ffffff)

![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=chaimae098&layout=compact&hide_border=true&title_color=c8522a&theme=default&text_color=333333&bg_color=ffffff)

![GitHub Streak](https://streak-stats.demolab.com?user=chaimae098&theme=default&hide_border=true&ring=c8522a&fire=c8522a&currStreakLabel=c8522a)

</div>

---

<div align="center">

```
╔══════════════════════════════════════════════╗
║  Open to Summer 2026 internships             ║
║  Anomaly Detection · MLOps · Backend / APIs  ║
╚══════════════════════════════════════════════╝
```

*Last updated: 2026 · ENSAM Casablanca*

</div>
