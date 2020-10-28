from typing import Dict

from fastapi import Depends, FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ArticleRequest(BaseModel):
    text: str


class ArticleResponse(BaseModel):
    confidence: Dict[str, float]
    article: Dict[str, int]

@app.get("/")
def index_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
def index(request: Request, title: str = Form(...), abstract: str = Form(...), model: Model = Depends(get_model)):

    result = []

    result.append(title)
    result.append(abstract)

    text = title + " [SEP] " + abstract
    article = model.predict(text)

    result.append(article)

    return templates.TemplateResponse("index.html", {"request": request, "result": result})


@app.post("/predict", response_model=ArticleResponse)
def predict(request: ArticleRequest, model: Model = Depends(get_model)):
    article = model.predict(request.text)

    return ArticleResponse(
        article=article
    )
