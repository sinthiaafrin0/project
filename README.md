# FastAPI + Next.js Machine Learning Q&A System

This project is a machine-learning-based question-answering system using **FastAPI (backend)** and **Next.js (frontend)**. Users can upload a CSV file to train a RandomForest model and then ask fixed questions via a dropdown menu to get predictions.

## Features
- **FastAPI Backend**: Handles CSV upload, model training, and predictions using RandomForest.
- **Next.js Frontend**: Allows users to upload CSV files and query the model using a dropdown menu.
- **Machine Learning**: Uses `scikit-learn` for model training.
- **Axios API Calls**: Fetch predictions from the backend.
- **CORS Handling**: Allows frontend-backend communication.

## Installation
### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **Node.js 16+**
- **npm or yarn**

### 1️⃣ Backend (FastAPI) Setup
#### Step 1: Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate    # Windows
```
#### Step 2: Install Dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas python-multipart
```
#### Step 3: Create `main.py`
```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_FILE = "model.pkl"

def train_model(file_path):
    df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns")
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Labels
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

@app.post("/learn")
def learn(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    try:
        train_model(file_path)
        os.remove(file_path)
        return {"message": "Model trained successfully!"}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))

@app.get("/ask")
def ask(q: str = Query(...)):
    if not os.path.exists(MODEL_FILE):
        raise HTTPException(status_code=400, detail="Model not found. Please train first.")
    model = joblib.load(MODEL_FILE)
    prediction = model.predict([[float(i) for i in q.split(',')]])[0]
    return {"prediction": prediction}
```
#### Step 4: Run the Backend
```bash
uvicorn main:app --reload
```
Check API at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2️⃣ Frontend (Next.js) Setup
#### Step 1: Create Next.js Project
```bash
npx create-next-app frontend
cd frontend
```
#### Step 2: Install Dependencies
```bash
npm install axios react-toastify
```
#### Step 3: Create `pages/upload.js`
```javascript
import { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

const Upload = () => {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    if (!file) return toast.error("Please select a file");
    const formData = new FormData();
    formData.append("file", file);
    try {
      await axios.post("http://127.0.0.1:8000/learn", formData);
      toast.success("Model trained successfully!");
    } catch (error) {
      toast.error("Upload failed");
    }
  };

  return (
    <div>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
};
export default Upload;
```
#### Step 4: Create `pages/query.js`
```javascript
import { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";

const questions = [
"Is he suffering from cancer?",
"Which stages of cancer?",
"Probability of his survival",
];

const Query = () => {
  const [selectedQuestion, setSelectedQuestion] = useState("");
  const [prediction, setPrediction] = useState("");

  const handleQuery = async () => {
    if (!selectedQuestion) return toast.error("Please select a question!");
    try {
      const response = await axios.get(`http://127.0.0.1:8000/ask?q=${selectedQuestion}`);
      setPrediction(response.data.prediction);
      toast.success(`Prediction: ${response.data.prediction}`);
    } catch (error) {
      toast.error("Error fetching prediction!");
    }
  };

  return (
    <div>
      <select onChange={(e) => setSelectedQuestion(e.target.value)}>
        <option value="">Select a question</option>
        {questions.map((q, i) => (
          <option key={i} value={q}>{q}</option>
        ))}
      </select>
      <button onClick={handleQuery}>Ask</button>
      <p>Prediction: {prediction}</p>
    </div>
  );
};
export default Query;
```
#### Step 5: Run the Frontend
```bash
npm run dev
```
Visit: [http://localhost:3000/upload](http://localhost:3000/upload)

---

## Troubleshooting
### **AxiosError: Network Error**
- Ensure **FastAPI is running**: `uvicorn main:app --reload`
- Check **CORS settings** in `main.py`
- Verify the API URL in `axios.get()`
- Check **DevTools > Network** in the browser

### **Model Not Found Error**
- Upload a CSV file at `/upload` before querying at `/query`

## License
This project is open-source. Feel free to modify and improve it!

