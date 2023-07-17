import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
# Load the saved model
model = joblib.load('cancer_model.pkl')

# Define the input data schema
class PredictionInput(BaseModel):
    GENDER: str
    AGE: int
    SMOKING: str
    YELLOW_FINGERS: str
    ANXIETY: str
    PEER_PRESSURE: str
    CHRONIC_DISEASE: str
    FATIGUE: str
    ALLERGY: str
    WHEEZING: str
    ALCOHOL_CONSUMING: str
    COUGHING: str
    SHORTNESS_OF_BREATH: str
    SWALLOWING_DIFFICULTY: str
    CHEST_PAIN: str

# Create the FastAPI application
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static files directory to serve the HTML and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the prediction route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_lung_cancer(data: PredictionInput):
    # Convert the input data to a dictionary
    input_data = data.dict()
    
    # Prepare the input features for prediction
    input_features = [input_data[col] for col in input_data]
    
    # Make the prediction
    prediction = model.predict([input_features])[0]
    
    # Return the prediction as a JSON response
    return {"prediction": prediction}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)