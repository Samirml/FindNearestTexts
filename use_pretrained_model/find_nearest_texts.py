import os
import numpy as np
import torch
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from src.data import load_and_preprocess_data
from src.config import config
from src.model_upgrade import TorchLSTM
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from use_pretrained_model.extract_keywords import extract_keywords
from src.train_utils import save_train_texts_and_keywords, save_test_texts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('/home/samir/PycharmProjects/FindNearestTexts/src/model_versions')
model_path = os.path.abspath('model_to_find_texts.pt')

# Load and preprocess data
tokenized_datasets, tokenizer, _, _, _ = load_and_preprocess_data(config["tokenizer_name"])

# Initialize the model
model = TorchLSTM(vocab_size=len(tokenizer),
                  hidden_size=config["hidden_size"],
                  num_layers=config["num_layers"],
                  bidirectional=True,
                  n_classes=config["n_classes"]).to(device)

# Load the saved model weights
model.load_state_dict(torch.load(model_path))

# Save texts and keywords for finding nearest texts
texts, keywords_per_text = save_train_texts_and_keywords(model, tokenizer, tokenized_datasets, device)
test_texts = save_test_texts(model, tokenizer, tokenized_datasets)

# Initialize the FastAPI application and Jinja2 templates
app = FastAPI()
templates = Jinja2Templates(directory="/home/samir/PycharmProjects/FindNearestTexts/templates")


class TextInput(BaseModel):
    """
    Data model to receive the input text and number of nearest texts (k).

    Args:
        text (str): Input text for which to find similar texts.
        k (int, optional): Number of nearest texts, defaults to 5.
    """
    text: str
    k: int = 5


def find_nearest_texts(model: torch.nn.Module, text, texts_train, keywords_per_text: list,
                       device: torch.device, k: int = 5, threshold: float = 0.2) -> tuple:
    """
    Finds the nearest texts to the input text based on Jaccard similarity of extracted keywords.

    Args:
        model (torch.nn.Module): Trained model for keyword extraction.
        text (str): The input text for which to find similar texts.
        texts_train (list): List of training texts.
        keywords_per_text (list): List of keywords for all training texts.
        device (torch.device): Device to perform inference on (CPU/GPU).
        k (int, optional): Number of nearest texts to return. Default is 5.
        threshold (float, optional): Minimum similarity threshold for recommendations. Default is 0.2.

    Returns:
        tuple: List of k nearest texts based on keyword similarity, a warning flag, and an error message if applicable.
    """
    # Check the length of the input text
    if len(text.strip()) < 10:  # For example, if less than 10 characters
        return [], True, "There is too little information. Please, complete the text!"

    # Extract keywords from the input text
    keywords = set(extract_keywords(text, device, model))
    scores = []

    # Calculate Jaccard similarity between input text keywords and training texts
    for another_text_keywords in keywords_per_text:
        jaccard_score = len(keywords & another_text_keywords) / len(keywords | another_text_keywords)
        scores.append(jaccard_score)

    # Find the indices of the k nearest texts
    nearest_idxs = np.argsort(scores)[::-1][:k]
    nearest_texts = [texts_train[i] for i in nearest_idxs]
    top_score = scores[nearest_idxs[0]]

    # Generate a warning if the top match score is below the threshold
    warning = top_score < threshold
    return nearest_texts, warning, None


@app.post("/find-nearest-texts/")
async def get_nearest_texts_form(request: Request, text: str = Form(...), k: int = Form(5)):
    """
    Endpoint for receiving text input from a form and returning the nearest texts.

    Args:
        request (Request): FastAPI request object.
        text (str): Input text from the form.
        k (int, optional): Number of nearest texts to find, defaults to 5.

    Returns:
        TemplateResponse: The response with the nearest texts or an error message.
    """
    try:
        nearest_texts, warning, error_message = find_nearest_texts(model, text, texts, keywords_per_text, device, k)

        # If there is an error message, show it on the web page
        if error_message:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error_message": error_message
            })

        # Return nearest texts and potential warning
        return templates.TemplateResponse("index.html", {
            "request": request,
            "nearest_texts": nearest_texts,
            "warning": warning
        })
    except Exception as e:
        # Handle any unexpected errors and return a 500 response
        return HTMLResponse(content=f"An error occurred: {str(e)}", status_code=500)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """
    Endpoint for rendering the main page of the application.

    Args:
        request (Request): FastAPI request object.

    Returns:
        TemplateResponse: Renders the main index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})