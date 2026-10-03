# Microsoft Fabric Data Science & AI

> A large, practical collection of Python notebooks and scripts for data science, machine learning, artificial intelligence, analytics, experimentation, optimization, and industry-focused use cases, organized for exploration in Microsoft Fabric and Python environments.

<p align="center">
  <strong>Learn by building. Explore by topic. Adapt to real-world problems.</strong>
</p>

---

## Overview

`microsoft-fabric-data-science-ai` is a hands-on repository built for people who want practical Python examples they can explore, adapt, and extend across the data science and AI lifecycle.

Rather than focusing on a single model family or business domain, the repository brings together a broad collection of technical patterns and applied use cases. Content is organized into topic folders so that you can move directly from an idea or problem area to relevant code.

The repository includes two complementary formats:

- **Jupyter notebooks** for interactive exploration and experimentation
- **Python scripts** for reusable, code-first implementations

The collection spans foundational statistics and machine learning, modern AI patterns, optimization, experimentation, responsible AI, data engineering, and a wide range of industry scenarios.

---

## Why this repository exists

Data science work rarely starts with a blank definition of an algorithm. It usually starts with a problem:

- How can demand be forecast more accurately?
- How can unusual behavior be detected?
- How can customers or products be segmented?
- How can limited resources be allocated more effectively?
- How can an AI system be evaluated or monitored?
- How can risk, reliability, quality, or operational performance be modeled?
- How can an analytical idea be translated into working Python quickly?

This repository is designed around that reality.

It provides a broad starter library of implementations that can be used for learning, prototyping, experimentation, technical demonstrations, and as foundations for more specialized solutions.

---

## Repository structure

```text
microsoft-fabric-data-science-ai/
|
|-- notebooks/                 Interactive Jupyter notebook implementations
|   |-- ai_agents/
|   |-- anomaly_detection/
|   |-- automl/
|   |-- causal_inference/
|   |-- clustering/
|   |-- computer_vision/
|   |-- data_engineering/
|   |-- forecasting/
|   |-- generative_ai/
|   |-- healthcare_analytics/
|   |-- optimization/
|   |-- responsible_ai/
|   |-- statistics/
|   `-- ...
|
|-- scripts/                   Python script implementations
|   |-- ai_agents/
|   |-- anomaly_detection/
|   |-- automl/
|   |-- causal_inference/
|   |-- data_engineering/
|   |-- forecasting/
|   |-- generative_ai/
|   |-- optimization/
|   |-- statistics/
|   `-- ...
|
|-- tools/                     Repository utilities and execution helpers
|-- catalog.json               Workflow metadata catalog
|-- requirements.txt           Core Python dependencies
|-- requirements-tested.txt    Tested dependency versions
|-- run_all.py                 Catalog-driven workflow runner
|-- run_all_scripts.py         Script execution utility
|-- validate_repository.py     Repository validation utility
|-- LICENSE
`-- README.md
```

The exact set of topic folders continues to grow as new examples are added.

---

## Explore by technical topic

The repository covers a wide spectrum of data science and AI concepts. Major areas include:

### Artificial intelligence

- AI agents and agentic workflows
- Generative AI
- Retrieval and prompt-oriented patterns
- AutoML
- Model evaluation
- Responsible AI
- Privacy engineering
- Recommendation systems

### Machine learning

- Classification
- Regression
- Clustering
- Anomaly detection
- Feature engineering
- Forecasting
- Time-series analysis
- Survival analysis
- Graph analytics
- Pattern mining

### Statistics and experimentation

- Bayesian methods
- Hypothesis testing
- Experimentation
- A/B testing
- Causal inference
- Uncertainty quantification
- Statistical diagnostics
- Signal processing

### Decision science

- Optimization
- Allocation algorithms
- Capacity planning
- Scheduling
- Simulation
- Resource allocation
- Multi-objective decision problems

### Data and engineering

- Data engineering
- Data governance
- Data quality
- Feature pipelines
- Developer analytics
- Observability-oriented use cases
- Semantic and metadata patterns

---

## Explore by industry and business domain

One of the goals of this repository is to connect technical methods to realistic business problems.

Examples span areas such as:

| Domain | Example themes |
| --- | --- |
| Healthcare | patient risk, care pathways, treatment, clinical operations |
| Financial services | portfolio analytics, payments, risk, collections, banking |
| Insurance | claims, underwriting, catastrophe and policy analytics |
| Retail | merchandising, availability, pricing and customer behavior |
| Marketing | attribution, campaigns, channel analytics and lift |
| Sales | opportunities, pipeline health, conversion and prioritization |
| Supply chain | inventory, procurement, cold chain and supplier analytics |
| Manufacturing | quality, reliability, production and equipment monitoring |
| Energy | grid analytics, renewable energy, demand and asset performance |
| Sustainability | emissions, circularity, waste and environmental impact |
| Aviation | baggage, fleet, crew and passenger analytics |
| Mobility | vehicles, charging, routing and transportation behavior |
| Telecommunications | network, subscriber, roaming and infrastructure analytics |
| Cybersecurity | access, identity, attack signals and security monitoring |
| Education | assessment, learner behavior, admissions and intervention |
| Legal | contracts, clauses, matters, intellectual property and compliance |
| Real estate | leasing, property, tenants, vacancy and zoning |
| Sports | performance, engagement, scouting and match analytics |
| Media | audience, content, creator and engagement analytics |
| Agriculture | crops, yield, sustainability and operational risk |
| Public sector | programs, infrastructure, benefits and civic use cases |

These are starter implementations, not production decision systems. Real deployments require domain validation, appropriate data governance, security controls, monitoring, and human oversight.

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/sahirmaharaj/microsoft-fabric-data-science-ai.git
cd microsoft-fabric-data-science-ai
```

### 2. Create a Python environment

Using a virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the core dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The core requirements are intentionally compact and include NumPy, pandas, Matplotlib, scikit-learn, and SciPy.

For the repository's tested environment, see:

```text
requirements-tested.txt
```

---

## Using the repository in Microsoft Fabric

The notebook collection is designed to be easy to adapt to a Microsoft Fabric data science workflow.

A typical workflow is:

1. Open your Microsoft Fabric workspace.
2. Create or open a Data Science notebook.
3. Locate a relevant example under `notebooks/`.
4. Bring the notebook logic into your Fabric environment.
5. Replace demonstration or generated data with your Lakehouse, Warehouse, or other governed data source.
6. Adapt feature engineering, modeling, evaluation, and output logic to your use case.
7. Validate the result against your organization's requirements before operational use.

The examples should be treated as starting points. Workspace configuration, runtime availability, package versions, security requirements, and data access patterns can differ between environments.

---

## How to find the right example

The easiest way to navigate the repository is to start with the problem rather than the algorithm.

### If you know the technique

Browse a technical category such as:

```text
notebooks/anomaly_detection/
notebooks/causal_inference/
notebooks/forecasting/
notebooks/optimization/
notebooks/statistics/
```

### If you know the business problem

Browse an applied category such as:

```text
notebooks/customer_analytics/
notebooks/energy_analytics/
notebooks/healthcare_analytics/
notebooks/marketing_analytics/
notebooks/supply_chain/
```

### Search from the command line

Find files containing a concept:

```bash
find notebooks scripts -type f | grep -i "forecast"
```

Search for multiple ideas:

```bash
find notebooks scripts -type f | grep -Ei "churn|retention|customer"
```

Search the source code itself:

```bash
grep -Rni "RandomForestClassifier" notebooks scripts
```

---

## Notebooks vs scripts

### `notebooks/`

Use notebooks when you want to:

- explore a technique interactively
- inspect intermediate results
- experiment with parameters
- learn the flow of an analytical solution
- adapt an example inside a notebook-oriented environment such as Microsoft Fabric

### `scripts/`

Use scripts when you want to:

- inspect a code-first implementation
- reuse logic in another Python project
- automate execution
- integrate an example into a larger workflow
- compare script and notebook approaches

The repository intentionally provides both styles because exploration and operationalization often require different workflows.

---

## Core Python stack

The lightweight core environment is based on:

```text
NumPy
pandas
Matplotlib
scikit-learn
SciPy
```

A tested environment is also recorded in `requirements-tested.txt` and includes notebook execution dependencies such as IPython, IPykernel, nbformat, and nbclient.

Individual workflows may require additional dependencies. Review the relevant code and environment before running an example.

---

## Workflow catalog

`catalog.json` provides structured metadata for a curated set of repository workflows.

Catalog entries can describe information such as:

- workflow ID
- category
- workflow name
- Python script path
- notebook path
- code size metadata
- dependencies

This makes it possible to programmatically discover and execute cataloged examples rather than treating the repository as an unstructured collection of files.

---

## Running cataloged workflows

`run_all.py` provides a catalog-driven execution utility with filtering, worker, timeout, and output controls.

List cataloged workflows:

```bash
python run_all.py --list
```

Run selected catalog entries:

```bash
python run_all.py --only <workflow-or-category>
```

Control concurrency and timeout:

```bash
python run_all.py --workers 2 --timeout 600
```

Choose an output location:

```bash
python run_all.py --output outputs
```

> **Note:** if repository files have been reorganized or renamed, catalog paths must be kept synchronized with their current locations before using catalog-driven execution.

---

## Repository validation

The repository includes `validate_repository.py` for validating cataloged workflow structure and, optionally, execution behavior.

The validation utility includes checks around notebook format, corresponding script/notebook source, structural constraints, and generated artifacts for cataloged workflows.

Static validation can be started with:

```bash
python validate_repository.py
```

Notebook execution can also be requested:

```bash
python validate_repository.py --execute-notebooks
```

The validator supports execution engines and configurable worker counts. As with the workflow runner, ensure `catalog.json` reflects current file paths before relying on catalog-based validation.

---

## Example learning paths

### Beginner: practical machine learning

```text
statistics
    -> feature engineering
    -> classification / regression
    -> model evaluation
    -> forecasting
```

Focus on understanding the data, establishing baselines, selecting meaningful metrics, and interpreting model behavior before increasing complexity.

### Intermediate: applied data science

```text
clustering
    -> anomaly detection
    -> recommendation systems
    -> time series
    -> optimization
```

At this level, explore how modeling choices interact with business objectives, constraints, and operational decisions.

### Advanced: modern AI systems

```text
generative AI
    -> AI agents
    -> evaluation
    -> responsible AI
    -> privacy engineering
    -> production-oriented monitoring
```

The goal is not only to build models, but to reason about the larger system around them.

### Decision intelligence

```text
forecasting
    -> uncertainty
    -> simulation
    -> optimization
    -> allocation
```

This path is useful when predictions ultimately need to support an action or decision.

---

## Example use-case journey

Suppose you are working on customer retention.

You could begin by exploring customer analytics examples to understand segmentation and retention signals. You might then move into classification to estimate churn risk, causal inference to reason about interventions, recommendation systems to personalize next-best actions, and experimentation to measure whether those interventions actually improve outcomes.

That progression reflects an important principle behind this repository:

> **A model is often only one component of a useful data science solution.**

The strongest solutions connect data preparation, modeling, evaluation, decision-making, experimentation, governance, and monitoring.

---

## Design principles

### Practical first

Examples are intended to make concepts tangible through executable Python rather than purely theoretical descriptions.

### Broad coverage

The collection intentionally spans techniques, industries, and business functions so that ideas can be transferred across domains.

### Discoverable structure

Files are grouped into descriptive topic folders rather than being stored as one large flat collection.

### Adaptable examples

The code is intended to be modified. Change the data, assumptions, parameters, metrics, models, and outputs to suit your problem.

### Responsible experimentation

A successful notebook run is not evidence that a model is ready for production. Evaluate data quality, bias, privacy, security, robustness, explainability, operational constraints, and real-world consequences before deployment.

---

## Working with your own data

A practical adaptation workflow is:

```text
Choose an example
        |
        v
Understand the expected inputs
        |
        v
Connect your data source
        |
        v
Map and validate the schema
        |
        v
Adapt preprocessing
        |
        v
Train or execute the analytical workflow
        |
        v
Evaluate with appropriate metrics
        |
        v
Validate assumptions and risks
        |
        v
Integrate outputs into your Fabric workflow
```

Avoid replacing sample data with production data and immediately trusting the resulting output. Changes in data distributions, definitions, missingness, class balance, leakage, time dependencies, and business context can materially change model behavior.

---

## Contributing

Contributions that improve the usefulness, correctness, organization, or breadth of the repository are welcome.

A useful contribution should ideally:

1. solve a clearly defined analytical problem
2. use a descriptive filename
3. live in the most appropriate topic folder
4. avoid unnecessary generated artifacts or large binary files
5. keep dependencies focused
6. be reproducible where practical
7. avoid committing secrets, credentials, tokens, or private data
8. include appropriate validation for the problem being solved

When adding a new example, prefer descriptive names such as:

```text
customer_churn_risk_model.py
inventory_demand_forecaster.ipynb
causal_treatment_effect_estimator.py
```

rather than generic names such as:

```text
model1.py
notebook_new.ipynb
test_final_v2.py
```

---

## Responsible use

The examples in this repository are educational and experimental starter implementations.

Before using an analytical or AI workflow in a real decision process, consider:

- data quality and representativeness
- privacy and confidentiality
- security and access control
- fairness and potential disparate impact
- model uncertainty
- explainability requirements
- monitoring and drift
- human review and escalation
- legal and regulatory requirements
- domain-specific validation

High-impact decisions should not be automated simply because a model can generate a prediction.

---

## Repository maintenance

As the collection grows, maintainers should preserve a few conventions:

- keep notebooks under `notebooks/<category>/`
- keep scripts under `scripts/<category>/`
- avoid loose notebooks and scripts at the category roots
- use lowercase `snake_case` filenames
- prefer descriptive names over numeric prefixes
- avoid duplicate files across categories
- update catalog metadata when cataloged files move
- keep dependency files current
- validate links and paths after reorganizing content

These conventions keep a large repository usable as it expands.

---

## License

This repository includes a `LICENSE` file. Review the license before redistributing or incorporating repository content into another project.

---

## Author

### Sahir Maharaj

Data Scientist · AI Engineer · Microsoft Fabric & Power BI Specialist

I build and share practical work across data science, artificial intelligence, analytics, and the Microsoft data ecosystem. This repository is intended to make a broad set of data science and AI patterns easier to discover, learn from, and adapt.

---

## Support the project

If this repository helps you:

- star the repository
- explore and adapt the examples
- share useful improvements
- open an issue when you find something that can be improved
- contribute new practical workflows

The goal is to keep building a useful, organized library for people working with Python, data science, AI, and Microsoft Fabric.

---

<p align="center">
  <strong>Microsoft Fabric · Python · Data Science · Machine Learning · AI · Analytics</strong>
</p>

<p align="center">
  Built and maintained by <strong>Sahir Maharaj</strong>
</p>
