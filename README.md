# Image Processing Tutorial Laboratory

Interactive Streamlit tutorial for common image-processing techniques.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py` and `requirements.txt` to a GitHub repository, then create a Streamlit Community Cloud app pointing to `app.py`.

## Notes
- Educational demonstration only, not a medical diagnostic tool.
- For scientific and medical datasets, retain originals and validate every preprocessing step.
- The HOG view is a simplified visualization, not a full production HOG descriptor.
