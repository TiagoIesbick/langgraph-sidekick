from twilio.rest import Client
from os import getenv
from langchain.agents import Tool


TWILIO_ACCOUNT_SID = getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = getenv("TWILIO_WHATSAPP_FROM")
TWILIO_WHATSAPP_TO = getenv("TWILIO_WHATSAPP_TO")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp(text: str) -> str:
    msg = client.messages.create(
        body=text,
        from_=f'whatsapp:{TWILIO_WHATSAPP_FROM}',
        to=f'whatsapp:{TWILIO_WHATSAPP_TO}'
    )
    return msg.sid

whatsapp_tool = Tool(
    name="send_whatsapp",
    func=send_whatsapp,
    description="Send a WhatsApp message via Twilio."
)
