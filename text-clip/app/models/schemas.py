from pydantic import BaseModel


class CheckResponse(BaseModel):
    result: str
    title: str = "mt-photos-ai-text-clip"
    help: str = "https://mtmt.tech/docs/advanced/ocr_api"


class TextClipRequest(BaseModel):
    text: str
