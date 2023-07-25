import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
# Load the saved model
model = joblib.load('cancer_model.pkl')
import sklearn
print(sklearn.__version__)
# Define the input data schema
class PredictionInput(BaseModel):
    gender: str = Form()
    age: int = Form()
    smoking: str = Form()
    yellowFingers: str = Form()
    anxiety: str = Form()
    peerPressure: str = Form()
    chronicDisease: str = Form()
    fatigue: str = Form()
    allergy: str = Form()
    wheezing: str = Form()
    alcoholConsuming: str = Form()
    coughing: str = Form()
    shortnessOfBreath: str = Form()
    swallowingDifficulty: str = Form()
    chestPain: str = Form()
    
    
# Create the FastAPI application
app = FastAPI()
templates = Jinja2Templates(directory="static")

# Mount the static files directory to serve the HTML and CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the prediction route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("/index.html", {"request": request})


@app.post("/predict")
async def predict_lung_cancer(
    request: Request,
    gender: str = Form(),
    age: int = Form(),
    smoking: str = Form(),
    yellowFingers: str = Form(),
    anxiety: str = Form(),
    peerPressure: str = Form(),
    chronicDisease: str = Form(),
    fatigue: str = Form(),
    allergy: str = Form(),
    wheezing: str = Form(),
    alcoholConsuming: str = Form(),
    coughing: str = Form(),
    shortnessOfBreath: str = Form(),
    swallowingDifficulty: str = Form(),
    chestPain: str = Form()
):
    # Convert the input data to a dictionary
    
    input_data = {
        # "gender":gender,
        # "age":float(age),
        # "smoking":float(smoking),
        "yellowFingers":float(yellowFingers),
        "anxiety":float(anxiety),
        "peerPressure":float(peerPressure),
        "chronicDisease":float(chronicDisease),
        "fatigue":float(fatigue),
        "allergy":float(allergy),
        "wheezing":float(wheezing),
        "alcoholConsuming":float(alcoholConsuming),
        "coughing":float(coughing),
        "shortnessOfBreath":float(shortnessOfBreath),
        "swallowingDifficulty":float(swallowingDifficulty),
        "chestPain":float(chestPain)
    }
    
    print(type(input_data))
    
    # Prepare the input features for prediction
    input_features = [input_data[col] for col in input_data]
    
    # Make the prediction
    prediction = model.predict([input_features])[0]
    

    
    # Return the prediction as a JSON response
    return {"prediction": int(prediction)}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)