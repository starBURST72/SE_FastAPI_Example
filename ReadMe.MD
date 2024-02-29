FastAPI Sentiment Analysis Service
This FastAPI service provides a simple API for sentiment analysis using the Hugging Face Transformers library.

Installation
To run the FastAPI service, you need to install the required dependencies. You can do this using pip:

bash
Copy code
pip install fastapi uvicorn transformers
Usage
Start the FastAPI service: Run the following command to start the FastAPI service:

bash
Copy code
uvicorn main:app --reload
Accessing the API: Once the service is running, you can access the API at http://127.0.0.1:8000/.

Endpoints
GET /: Returns a message indicating that the FastAPI service has started.

GET /{text}: Performs sentiment analysis on the provided text.

Parameters:

text (str): The text for sentiment analysis.
Response:

label (str): The predicted sentiment label (e.g., "positive", "negative", "neutral").
score (float): The confidence score for the predicted sentiment.
POST /predict/: Performs sentiment analysis on the text provided in the request body.

Request Body:

json
Copy code
{
    "text": "The text to analyze."
}
Response:

label (str): The predicted sentiment label (e.g., "positive", "negative", "neutral").
score (float): The confidence score for the predicted sentiment.
Example
bash
Copy code
# Start the FastAPI service
uvicorn main:app --reload
bash
Copy code
# Send a GET request to perform sentiment analysis
curl -X 'GET' \
  'http://127.0.0.1:8000/This%20is%20a%20great%20movie!' \
  -H 'accept: application/json'
bash
Copy code
# Send a POST request to perform sentiment analysis
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "This is a great movie!"
}'
