import requests

MYMEMORY_URL = "https://api.mymemory.translated.net/get"

def translate(text: str) -> str:
    try:
        response = requests.get(
            MYMEMORY_URL,
            params={"q": text, "langpair": "ar|de"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return data["responseData"]["translatedText"]
    except Exception:
        return text
