# SHUDDHOTA Framework: Quantitative Evaluation Pipeline
**Master of Science in Computer and Communication Engineering**  
**Patuakhali Science and Technology University**  

This repository contains the reproducible simulation and analytical pipeline for the SHUDDHOTA Framework, designed to mitigate AI-generated synthetic media threats in Bangladesh. 

### Files Included:
* `create_data.py`: Generates the n=100 simulated proof-of-concept dataset (Seed: 42).
* `analysis.py`: Evaluates H1-H8, calculates 14 classification metrics, and exports thesis figures.
* `app.py`: Deploys the live Streamlit dashboard for real-time framework interaction.
* `my_real_survey_data.csv`: The generated baseline dataset.

### How to Run Locally:
1. `pip install -r requirements.txt`
2. `python analysis.py` (to generate static thesis artifacts)
3. `streamlit run app.py` (to launch the interactive dashboard)
