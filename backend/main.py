from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ FIX CORS HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):
    values = data.get("values", [])

    # simple mock logic (replace with your model)
    prediction = sum(values) * 0.001

    return {"prediction": prediction}
