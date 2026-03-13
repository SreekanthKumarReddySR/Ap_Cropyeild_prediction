# AP_states_cropyield_prediction

Crop yield prediction for crops across Andhra Pradesh districts using a custom random forest regressor and a Flask web app.

## Files

```text
app.py
Model3.py
final_data.csv
enhanced_random_forest_regressor.pkl
label_encoders.pkl
requirements.txt
.python-version
templates/
```

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## Render

Recommended Render settings:

- Root Directory: leave empty if this repository root contains `app.py`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

The repo pins Python with `.python-version` set to `3.11`.

## CI/CD

The repository includes GitHub Actions workflow `.github/workflows/ci-cd.yml`.

CI behavior:

- Runs on push to `main`
- Runs on pull requests targeting `main`
- Installs dependencies
- Smoke-tests the Flask app by importing `app.py` and calling `/`

CD behavior:

- After CI passes on `main`, GitHub Actions can trigger a Render deploy hook

To enable deploy from GitHub Actions:

1. Open your Render web service.
2. Copy the Deploy Hook URL.
3. In GitHub, add repository secret `RENDER_DEPLOY_HOOK_URL`.

If the secret is not set, the deploy job is skipped and CI still runs.
