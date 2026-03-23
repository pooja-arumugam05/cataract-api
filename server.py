
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from einops import rearrange, repeat
from PIL import Image
import io, os, numpy as np

app = FastAPI(title="CataractAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DEVICE = torch.device("cpu")
CLASSES = ["normal", "mild", "moderate", "severe"]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class MHSA(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.h=h; self.dk=d//h; self.sc=self.dk**-0.5
        self.qkv=nn.Linear(d,d*3,bias=False)
        self.proj=nn.Linear(d,d)
        self.ad=nn.Dropout(drop); self.pd=nn.Dropout(drop)
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,self.dk).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        attn=self.ad((q@k.transpose(-2,-1))*self.sc).softmax(dim=-1)
        x=(attn@v).transpose(1,2).reshape(B,N,C)
        return self.pd(self.proj(x)), attn

class TBlock(nn.Module):
    def __init__(self, d, h, ffn, drop=0.1):
        super().__init__()
        self.n1=nn.LayerNorm(d); self.attn=MHSA(d,h,drop)
        self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,ffn),nn.GELU(),
                               nn.Dropout(drop),nn.Linear(ffn,d),nn.Dropout(drop))
    def forward(self, x):
        a,w=self.attn(self.n1(x)); x=x+a; x=x+self.ffn(self.n2(x))
        return x,w

class CataractViT(nn.Module):
    def __init__(self, num_classes=4, d=512, heads=8, layers=4, ffn=1024, drop=0.1):
        super().__init__()
        bb=models.resnet50(weights=None)
        self.cnn=nn.Sequential(*list(bb.children())[:-2])
        self.proj=nn.Sequential(nn.Conv2d(2048,d,1),nn.BatchNorm2d(d),nn.GELU())
        self.cls=nn.Parameter(torch.randn(1,1,d)*0.02)
        self.pos=nn.Parameter(torch.randn(1,50,d)*0.02)
        self.drop=nn.Dropout(drop)
        self.blocks=nn.ModuleList([TBlock(d,heads,ffn,drop) for _ in range(layers)])
        self.norm=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(d,num_classes))
    def forward(self, x):
        B=x.shape[0]
        f=self.cnn(x); f=self.proj(f)
        f=rearrange(f,"b c h w -> b (h w) c")
        cls=repeat(self.cls,"1 1 d -> b 1 d",b=B)
        t=self.drop(torch.cat([cls,f],1)+self.pos)
        for blk in self.blocks: t,_=blk(t)
        return self.head(self.norm(t)[:,0])

model = None

def load_model():
    global model
    if model is not None:
        return
    import gdown
    if not os.path.exists("model.pth"):
        print("Downloading model from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={os.environ.get('MODEL_FILE_ID','')}",
            "model.pth", quiet=False)
    m = CataractViT(num_classes=4).to(DEVICE)
    ckpt = torch.load("model.pth", map_location=DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    model = m
    print("Model loaded.")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model": "CataractViT", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        load_model()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
        idx  = int(probs.argmax())
        conf = float(probs[idx]) * 100
        sev  = (float(probs[1])*33 +
                float(probs[2])*66 +
                float(probs[3])*100)
        return {
            "predicted_class":  CLASSES[idx],
            "confidence":       round(conf, 2),
            "severity_percent": round(sev, 2),
            "probabilities": {
                c: round(float(p)*100, 2)
                for c, p in zip(CLASSES, probs)
            },
            "low_confidence": conf < 70.0
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
