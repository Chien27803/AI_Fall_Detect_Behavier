import requests

TELEGRAM_BOT_TOKEN = "8639607585:AAG7_lj5qkPOE6jarwBZOADdtZjzkLJX7XQ"
TELEGRAM_CHAT_ID = "8697469060"

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
data = {
    "chat_id": TELEGRAM_CHAT_ID,
    "text": "Test Telegram OK"
}

r = requests.post(url, data=data, timeout=10)
print(r.status_code)
print(r.text)