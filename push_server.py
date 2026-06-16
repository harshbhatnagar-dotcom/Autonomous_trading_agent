import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
import resend

load_dotenv(override=True)

NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL")
resend.api_key = os.getenv("RESEND_API_KEY")


mcp = FastMCP("push_server")


class PushModelArgs(BaseModel):
    message: str = Field(description="A email html body to push")


@mcp.tool()
def send_email(body:PushModelArgs):
    """ Send out an email with the given body to all sales prospects """
    try:
        response = resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": NOTIFICATION_EMAIL,
            "subject":"Market Seen",
            "html": body.message
        })
        print("Resend response:", response)
        return {"status":"success"}

    except Exception as e:
        print("Resend error:", e)
        return False


if __name__ == "__main__":
    mcp.run(transport="stdio")