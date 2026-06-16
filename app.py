import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from openai import AsyncOpenAI
from accounts import Account
import json

from agents import (
    Agent,
    Runner,
    Tool,
    trace,
    OpenAIChatCompletionsModel,
)

from agents.mcp import MCPServerStdio

from accounts_client import (
    read_accounts_resource,
    read_strategy_resource,
)

load_dotenv(override=True)

# ============================================================
# ENV
# ============================================================

polygon_api_key = os.getenv("POLYGON_API_KEY")

market_mcp = {
    "command": "uv",
    "args": ["run", "market_server.py"]
}

trader_mcp_server_params = [
    {"command": "uv", "args": ["run", "accounts_server.py"]},
    {"command": "uv", "args": ["run", "push_server.py"]},
    market_mcp,
]

tavily_env = {
    "TAVILY_API_KEY": os.getenv("TAVILY_API")
}

researcher_mcp_server_params = [
    {"command": "uvx", "args": ["mcp-server-fetch"]},
    {
        "command": "npx",
        "args": ["-y", "tavily-mcp@latest"],
        "env": tavily_env,
    },
]

# ============================================================
# MCP SERVERS
# ============================================================

researcher_mcp_servers = [
    MCPServerStdio(
        params=params,
        client_session_timeout_seconds=30
    )
    for params in researcher_mcp_server_params
]

trader_mcp_servers = [
    MCPServerStdio(
        params=params,
        client_session_timeout_seconds=30
    )
    for params in trader_mcp_server_params
]

mcp_servers = trader_mcp_servers + researcher_mcp_servers

# ============================================================
# MODELS
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

groq_client = AsyncOpenAI(
    base_url=GROQ_BASE_URL,
    api_key=groq_api_key,
)

gpt_oss_120 = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-120b",
    openai_client=groq_client,
)

gpt_oss_20 = OpenAIChatCompletionsModel(
    model="openai/gpt-oss-20b",
    openai_client=groq_client,
)

llama = OpenAIChatCompletionsModel(
    model="llama-3.3-70b-versatile",
    openai_client=groq_client,
)

scout = OpenAIChatCompletionsModel(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    openai_client=groq_client,
)

# ============================================================
# RESEARCHER
# ============================================================

async def get_researcher(mcp_servers) -> Agent:
    instructions = f"""You are a financial researcher. You are able to search the web for interesting financial news,
    look for the possible trading opportunities, and help with research.
    Based on the request, you carry out the necessary research and repond with your findings.
    Take time to make multiple searches to get a comprehensive view, and then summarize your findings.
    If there in not a specific request , then just respond with the investment opportunities based on searching latest news.
    The current datetime is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    RESEARCH RULES:
- Use as few searches as necessary to answer the question accurately.
- Start with broad searches and only perform follow-up searches when needed.
- Never repeatedly search for the same information.
- Limit yourself to a maximum of 5-8 searches for broad research tasks.
- For specific questions, prefer 2-4 focused searches.
- Read only the most relevant sources.
- Ignore duplicate articles covering the same event.
- Summarize information immediately instead of retaining large amounts of raw text.
- Never copy long passages from articles.
- Extract only key facts, numbers, dates, catalysts, risks, and conclusions.
- Keep intermediate reasoning concise.
- When enough evidence has been collected, stop searching and produce the answer.
    """

    researcher = Agent(
        name="Researcher",
        instructions=instructions,
        model="gpt-4o-mini",
        mcp_servers=mcp_servers,
    )

    return researcher


async def get_researcher_tool(mcp_servers) -> Tool:
    researcher = await get_researcher(mcp_servers)

    return researcher.as_tool(
        tool_name="Researcher",
        tool_description="""
        This tool researches online for news and opportunities,
        either based on your specific request to look into a certain stock,
        or generally for notable financial news and opportunities.
        Describe what kind of research are you looking for.
        """,
    )

# ============================================================
# TRADER CONFIG
# ============================================================

agent_name = "Harsh"
harsh_initial_strategy="You are a day trader who aggressively buys and sells shares based on the news and market conditions."
Account.get("Harsh").reset(harsh_initial_strategy)

prompt = """
Use your tools to make decisions about your portfolio.
Investigate the news and the market, make your decision, make the trades, and respond with a summary of your actions.
"""

servers_connected = False

# ============================================================
# CONNECT MCP
# ============================================================

async def connect_servers():
    global servers_connected

    if servers_connected:
        return

    for server in mcp_servers:
        await server.connect()

    servers_connected = True

# ============================================================
# MAIN EXECUTION
# ============================================================

async def run_trader():

    await connect_servers()

    account_details = await read_accounts_resource(agent_name)
    strategy = await read_strategy_resource(agent_name)

    instructions = f"""
You are a trader that manages a portfolio of shares. Your name is {agent_name} and your account is under your name, {agent_name}.
You have access to tools that allow you to search the internet for company news, check stock prices, and buy and sell shares.
Your investment strategy for your portfolio is:
{strategy}
Your current holdings and balance is:
{account_details}
You have the tools to perform a websearch for relevant news and information.
You have tools to check stock prices.
You have tools to buy and sell shares.
You have tools to save memory of companies, research and thinking so far.
Please make use of these tools to manage your portfolio. Carry out trades as you see fit; do not wait for instructions or ask for confirmation.
"""

    researcher_tool = await get_researcher_tool(
        researcher_mcp_servers
    )

    trader = Agent(
        name=agent_name,
        instructions=instructions,
        tools=[researcher_tool],
        mcp_servers=trader_mcp_servers,
        model="gpt-4o-mini",
    )

    with trace(agent_name):
        result = await Runner.run(
            trader,
            prompt,
            max_turns=20,
        )

    return result.final_output

# ============================================================
# GRADIO
# ============================================================



# ============================================================
# ACCOUNT REPORT
# ============================================================

async def get_account_report():
    try:
        strategy = await read_strategy_resource(agent_name)

        raw = await read_accounts_resource(agent_name)
        account = json.loads(raw)

        holdings = account.get("holdings", {})
        transactions = account.get("transactions", [])

        holdings_text = "\n".join(
            f"• {symbol}: {qty} shares"
            for symbol, qty in holdings.items()
        ) or "No holdings"

        recent_transactions = "\n".join(
            f"• {t['symbol']} | {t['quantity']} @ ${t['price']:.2f}"
            for t in transactions[-5:]
        ) or "No transactions"

        return f"""
📊 ACCOUNT REPORT
Trader: {account['name'].title()}
💰 Balance
${account['balance']:,.2f}
📈 Total Portfolio Value
${account['total_portfolio_value']:,.2f}
📉 Profit / Loss
${account['total_profit_loss']:,.2f}
🎯 Strategy
{strategy}
🏦 Holdings
{holdings_text}
📝 Recent Transactions
{recent_transactions}
"""

    except Exception as e:
        return f"Error loading account report:\n\n{e}"

async def execute():
    try:
        result = await run_trader()
        return result
    except Exception as e:
        return f"Error:\\n\\n{str(e)}"


# ============================================================
# UI
# ============================================================

with gr.Blocks(
    title="Autonomous Trading Agent"
) as ui:

    gr.Markdown("# 📈 Autonomous Trading Agent")

    with gr.Row():

        account_report = gr.Textbox(
            label="📊 Account Report",
            lines=25,
            interactive=False
        )

        trader_output = gr.Textbox(
            label="Trader Output",
            lines=30,
            interactive=False
        )

    with gr.Row():

        refresh_btn = gr.Button("🔄 Refresh Report")

        run_btn = gr.Button("🚀 Run Trader")

    refresh_btn.click(
        fn=get_account_report,
        outputs=account_report
    )

    run_btn.click(
        fn=execute,
        outputs=trader_output
    )

    run_btn.click(
        fn=get_account_report,
        outputs=account_report
    )

    ui.load(
        fn=get_account_report,
        outputs=account_report
    )


if __name__ == "__main__":
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    ) 
