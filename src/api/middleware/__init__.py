from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-poweros-frontend-76amkqlnx-riyas-projects-6e47c769.vercel.app",
        "https://*.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)