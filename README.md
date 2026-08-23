# FoodSight

Streamlit app for uploading or capturing a food image.

### Run locally

1. Create and set up the virtual environment.

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Start the app.

   ```bash
   .venv/bin/streamlit run app.py --server.address=0.0.0.0 --server.port=8501
   ```
