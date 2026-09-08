# ScrollIntel

**Multi-Agent AI Engineering & Data Decision-Support Platform**

ScrollIntel is an engineering project exploring how specialized AI agents can coordinate data analysis, machine-learning workflows, quality checks, reporting and technical decision support through a common platform.

The project should be understood as an **AI-assisted engineering system**, not as a replacement for accountable technical leadership, domain experts or data-science teams. Human review remains necessary for consequential technical, business and policy decisions.

## What the repository demonstrates

### Multi-agent workflow design

The platform includes specialized agent roles for areas such as:

- technical architecture and planning;
- data analysis;
- machine-learning workflows;
- AI engineering;
- business analysis and reporting;
- software quality and validation.

These roles organize tasks and tools. They do not establish that an agent has the judgment, accountability or organizational authority of a human executive.

### Data and ML capabilities

- CSV, Excel, JSON and Parquet processing;
- automated data profiling and exploratory analysis;
- machine-learning workflow orchestration;
- interactive visualisation and reporting;
- natural-language interaction with analytical workflows;
- system health and performance monitoring;
- test and quality-assurance tooling.

### Architecture

**Backend:** Python · FastAPI · SQLAlchemy · PostgreSQL

**Frontend:** Next.js · React · TypeScript · Tailwind CSS

**Data/ML:** pandas · NumPy · scikit-learn · AI model integrations

**Infrastructure:** Docker · Redis · Nginx · Prometheus/Grafana-oriented monitoring

## Engineering workflow

```text
User / Dataset / Task
        ↓
Task decomposition
        ↓
Specialized agent/tool routing
        ↓
Data validation and analysis
        ↓
Model / report / technical output
        ↓
Quality checks
        ↓
Human review and decision
```

## Potential public-interest relevance

The underlying architecture could be adapted to data-intensive research and development workflows where analysts need repeatable data preparation, modelling, quality checks and reporting. Examples include programme analytics, socioeconomic datasets, operational monitoring and research support.

These are potential applications of the architecture, **not claims of deployment by governments, UN agencies or development organisations**.

## Evidence boundaries

This repository contains substantial engineering and experimentation, but feature presence does not by itself prove production reliability, model validity, organizational impact or return on investment.

Accordingly:

- no customer testimonials are presented without verifiable evidence;
- no claim is made that ScrollIntel replaces a CTO, consultant or data-science team;
- model outputs require validation for the relevant dataset and use case;
- production readiness should be established through current test, security and deployment evidence rather than README language.

## Quick start

### Docker

```bash
python scripts/setup-environment.py
docker-compose up -d
python scripts/health-check.py
```

### Development

```bash
pip install -r requirements.txt
python init_database.py
uvicorn scrollintel.api.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Testing

```bash
pytest tests/
```

Additional repository checks include frontend tests, end-to-end validation and production-readiness tooling. A passing script should be treated as evidence for the conditions it actually tests, not as a blanket certification of the whole system.

## Security and configuration

Secrets and deployment credentials must not be committed to source control. The repository is undergoing credential-hygiene remediation for legacy tracked environment files. Current or historical credentials should be treated as potentially exposed until they have been independently rotated/revoked where applicable.

`.gitignore` defines local environment and credential files that should remain outside version control.

## Documentation

- `INSTALLATION_GUIDE.md`
- `QUICK_START_GUIDE.md`
- `docs/DEPLOYMENT.md`
- `docs/TROUBLESHOOTING.md`
- local FastAPI documentation at `/docs` when the API is running

## Project direction

The current goal is to make the system more defensible as an engineering portfolio project by improving:

- reproducible evaluation;
- security and credential hygiene;
- model-quality evidence;
- data provenance;
- human-review controls;
- reliable deployment and observability;
- documentation that separates implemented features from aspirations.

## License

See `LICENSE` for repository licensing terms.
