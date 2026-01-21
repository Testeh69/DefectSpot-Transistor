from fastapi import FastAPI, File, UploadFile
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from models.models import AutoEncodeur
import io

app = FastAPI()


model = AutoEncodeur(in_ch=1, latent_space=100)
model.load_state_dict(torch.load("models/Def_Trans_5gen.pth", map_location="cpu"))
model.eval()

transform = Compose([
    Resize((256, 256)),
    ToTensor(),
])


@app.post("/file")
async def get_pictures(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    tensor_img = transform(image)
    tensor_img = tensor_img.unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        result = model(tensor_img)
    predicted_mask = torch.abs(result.squeeze() - tensor_img.squeeze()).numpy()
    return {
        "result": predicted_mask.tolist()
    }



#python -m uvicorn api.main:app --reload