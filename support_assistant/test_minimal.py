from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Test API",
    description="Minimal test API",
    version="1.0.0"
)

class Item(BaseModel):
    name: str
    price: float

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items/")
async def create_item(item: Item):
    return item

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7862)