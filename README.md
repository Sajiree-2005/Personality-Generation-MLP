# Text-Based Personality Generation using an MLP

A lightweight web application that uses a previously trained Multi-Layer Perceptron (MLP) to estimate five Big Five personality trait scores from 50 questionnaire responses and generate a short, readable trait overview.

The app loads the existing model artifacts. It **does not train or retrain the model** when deployed.

## Features

- 50 statements based on the questionnaire items used by the trained model.
- Five response options per statement, from 1 (Disagree) to 5 (Agree).
- Input validation to ensure all 50 responses are provided.
- Predictions for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.
- Scores clipped to the model project's 1–5 scale.
- Low, Medium, and High labels using the thresholds defined in the notebook.
- A template-based text overview based on each predicted score.
- Responsive layout, progress count, and print/save results option.
- `/health` endpoint for a simple deployment health check.

## How the prediction works

1. The user submits 50 responses, each from 1 to 5.
2. The app arranges the answers in the exact feature order saved in the model metadata.
3. The saved `StandardScaler` transforms the input using the fitted training-set statistics.
4. The trained Keras MLP predicts five continuous trait scores.
5. The predictions are limited to the 1–5 range and mapped to Low, Medium, or High:
   - **Low:** score below 2.5
   - **Medium:** score from 2.5 to below 3.5
   - **High:** score of 3.5 or above
6. The matching descriptions from the saved metadata are shown with the scores.

The app preserves the preprocessing and feature order associated with the supplied model. It does not read the training dataset, run exploratory analysis, compare models, or execute the notebook during inference.

## Project structure

```text
Personality-Generation-MLP/
├── app.py
├── requirements.txt
├── render.yaml
├── README.md
├── templates/
│   └── index.html
└── output/
    ├── personality_mlp_model.keras
    ├── personality_input_scaler.pkl
    └── personality_model_metadata.pkl
```

The deployment only needs the three model artifacts listed above. The notebook and metric CSV files are not required to serve predictions. If they already exist in your GitHub repository, you can leave them there; the web service does not execute them.

## Run locally

### 1. Requirements

- Python 3.10 or 3.11 recommended.
- Git, if you plan to deploy from GitHub.
- The model artifacts in the `output/` folder.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app.py
```

Open `http://127.0.0.1:10000` in your browser. Submit the questionnaire to see the results. The health endpoint is available at `http://127.0.0.1:10000/health`.

## Deploy on Render

### Option A: Use the included `render.yaml`

1. Add `app.py`, `requirements.txt`, `render.yaml`, `templates/index.html`, and the required three artifacts under `output/` to the root of your GitHub repository.
2. Commit and push the changes.
3. Sign in to [Render](https://render.com/).
4. Choose **New + → Blueprint** and connect the repository containing `render.yaml`.
5. Review the service settings and create the service.
6. Wait for the build and deployment to complete.
7. Open the generated `onrender.com` URL and test the questionnaire.
8. Visit `/health` on the same domain. A healthy response should include `"status": "ok"` and `"model_loaded": true`.

### Option B: Create a Web Service manually

1. In Render, choose **New + → Web Service**.
2. Connect the GitHub repository.
3. Set **Runtime** to `Python`.
4. Set **Build Command** to:

   ```bash
   pip install -r requirements.txt
   ```

5. Set **Start Command** to:

   ```bash
   gunicorn app:app
   ```

6. Select the free instance if it is available for your account and region.
7. Deploy and test the public URL.

Render's free web services may sleep after inactivity and take some time to wake up on the next request. Build times and free-tier limits can change.

## Updating the app

After making changes locally, commit and push them to the connected GitHub branch. Render can redeploy when it detects new commits, depending on the service's auto-deploy settings.

Do not change the feature order, scaler, or model independently: they must remain consistent with one another.

## Troubleshooting

### `FileNotFoundError` for a model artifact

Confirm that all three files exist under `output/` with exactly these names:

- `personality_mlp_model.keras`
- `personality_input_scaler.pkl`
- `personality_model_metadata.pkl`

Paths and filename capitalization matter on Render.

### Build fails while installing TensorFlow

Check the Python runtime and the current package compatibility shown in the build log. The provided dependency versions are a starting point for a compatible Python environment; if Render changes its default runtime, explicitly select a supported Python version in the service settings or adjust the dependency pins together.

### The service starts but predictions fail

Check the Render logs. The model, scaler, and metadata must be the matching artifacts saved with the trained model. Do not replace the scaler with a newly fitted scaler.

### The site takes a while to open

This can happen with a free service after a period without traffic. Wait for the service to wake up, then retry.

## Responsible use and limitations

This project is an educational demonstration of multi-output regression and text-based trait summaries. It is not a clinical instrument, a diagnosis, or a validated hiring or decision-making tool. Personality scores are sensitive to response patterns and to the data and assumptions used to train the model.

In this project, the notebook's target trait scores are calculated from the same 50 responses supplied to the model. Therefore, evaluation metrics measure how well the model reconstructs those calculated scores; they do not independently prove psychological accuracy. The generated text is a template selected from the predicted score levels, not free-form psychological analysis.

The app does not store questionnaire submissions in a database. Responses are processed to generate the current result; avoid submitting information you do not want processed by the hosted service.

## License

Add a license file if you intend to distribute or reuse this project publicly. Confirm the dataset and questionnaire licensing terms before redistribution.
