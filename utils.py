from google import genai
from google.genai import types
import pandas as pd
import json
import duckdb
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import os
import sqlite3

import random
# Set Phoenix ports to random values to avoid Address in Use errors from stuck processes
os.environ["PHOENIX_GRPC_PORT"] = str(random.randint(4000, 5000))
os.environ["PHOENIX_PORT"] = str(random.randint(6000, 7000))

from helper import get_gemini_api_key, get_phoenix_endpoint

import phoenix as px
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode

import warnings
warnings.filterwarnings('ignore')

# initialize the Gemini client
gemini_api_key = get_gemini_api_key()
client = genai.Client(api_key=gemini_api_key)

MODEL = "gemini-2.5-flash"

PROJECT_NAME = "evaluating-agent"

# Launch the Phoenix server locally on a random open port to avoid conflicts
try:
    session = px.launch_app(host="127.0.0.1")
    phoenix_endpoint = session.url
except Exception as e:
    print(f"Warning: Failed to launch Phoenix server: {e}")
    phoenix_endpoint = get_phoenix_endpoint() or "http://localhost:6006/"
    session = None

if not phoenix_endpoint.endswith("/"):
    phoenix_endpoint += "/"

tracer_provider = register(
    project_name=PROJECT_NAME,
    endpoint=phoenix_endpoint + "v1/traces"
)

tracer = tracer_provider.get_tracer(__name__)

# ## Defining the tools
TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'

SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """Generate an SQL query based on a prompt"""
    formatted_prompt = SQL_GENERATION_PROMPT.format(prompt=prompt, 
                                                    columns=columns, 
                                                    table_name=table_name)

    with tracer.start_as_current_span("generate_sql_query_llm", openinference_span_kind="llm") as span:
        span.set_input(value=formatted_prompt)
        response = client.models.generate_content(
            model=MODEL,
            contents=formatted_prompt,
        )
        span.set_output(value=response.text)
        span.set_status(StatusCode.OK)
    
    return response.text

@tracer.tool()
def lookup_sales_data(prompt: str) -> str:
    """Implementation of sales data lookup from parquet file using SQL"""
    try:
        table_name = "sales"
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        duckdb.sql(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        
        sql_query = generate_sql_query(prompt, df.columns, table_name)
        sql_query = sql_query.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "")

        with tracer.start_as_current_span(
            "execute_sql_query", 
            openinference_span_kind="chain"
        ) as span:
            span.set_input(sql_query)
            result = duckdb.sql(sql_query).df()
            span.set_output(value=str(result))
            span.set_status(StatusCode.OK)
        
        return result.to_string()
    except Exception as e:
        return f"Error accessing data: {str(e)}"

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

@tracer.tool()
def analyze_sales_data(prompt: str, data: str) -> str:
    """Implementation of AI-powered sales data analysis"""
    formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

    with tracer.start_as_current_span("analyze_sales_data_llm", openinference_span_kind="llm") as span:
        span.set_input(value=formatted_prompt)
        response = client.models.generate_content(
            model=MODEL,
            contents=formatted_prompt,
        )
        span.set_output(value=response.text)
        span.set_status(StatusCode.OK)
    
    analysis = response.text
    return analysis if analysis else "No analysis could be generated"

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")

@tracer.chain()
def extract_chart_config(data: str, visualization_goal: str) -> dict:
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(data=data, visualization_goal=visualization_goal)
    
    with tracer.start_as_current_span("extract_chart_config_llm", openinference_span_kind="llm") as span:
        span.set_input(value=formatted_prompt)
        response = client.models.generate_content(
            model=MODEL,
            contents=formatted_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisualizationConfig,
            ),
        )
        span.set_output(value=response.text)
        span.set_status(StatusCode.OK)
    
    try:
        content = json.loads(response.text)
        return {
            "chart_type": content.get("chart_type", "line"),
            "x_axis": content.get("x_axis", "date"),
            "y_axis": content.get("y_axis", "value"),
            "title": content.get("title", visualization_goal),
            "data": data
        }
    except Exception:
        return {
            "chart_type": "line", 
            "x_axis": "date",
            "y_axis": "value",
            "title": visualization_goal,
            "data": data
        }

CREATE_CHART_PROMPT = r"""
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
IMPORTANT: Use matplotlib.pyplot for charting. Do not use plt.show(), as this runs in a terminal without a GUI display. Instead, save the chart to a file named 'chart.png' in the current directory using plt.savefig('chart.png') and print a success message.
If you use regular expressions in strings (like '\s+'), ensure you use raw strings (e.g. r'\s+') to avoid SyntaxWarnings.
config: {config}
"""

@tracer.chain()
def create_chart(config: dict) -> str:
    formatted_prompt = CREATE_CHART_PROMPT.format(config=config)
    
    with tracer.start_as_current_span("create_chart_llm", openinference_span_kind="llm") as span:
        span.set_input(value=formatted_prompt)
        response = client.models.generate_content(
            model=MODEL,
            contents=formatted_prompt,
        )
        span.set_output(value=response.text)
        span.set_status(StatusCode.OK)
    
    code = response.text
    code = code.replace("```python", "").replace("```", "")
    code = code.strip()
    return code

@tracer.tool()
def generate_visualization(data: str, visualization_goal: str) -> str:
    config = extract_chart_config(data, visualization_goal)
    code = create_chart(config)
    try:
        exec(code, globals())
        return f"Chart successfully generated and saved to chart.png.\n\nCode used:\n{code}"
    except Exception as e:
        return f"Error executing chart code: {str(e)}\n\nCode used:\n{code}"

tool_schemas = [
    {
        "name": "lookup_sales_data",
        "description": "Implementation of sales data lookup from parquet file using SQL",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt."}
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "analyze_sales_data", 
        "description": "Implementation of AI-powered sales data analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "data": {"type": "string"}
            },
            "required": ["data", "prompt"]
        }
    },
    {
        "name": "generate_visualization",
        "description": "Generate Python code to create data visualizations",
        "parameters": {
            "type": "object", 
            "properties": {
                "data": {"type": "string"},
                "visualization_goal": {"type": "string"}
            },
            "required": ["data", "visualization_goal"]
        }
    }
]

tools_list = [types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name=schema["name"],
            description=schema["description"],
            parameters=schema.get("parameters")
        ) for schema in tool_schemas
    ]
)]

tool_implementations = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data, 
    "generate_visualization": generate_visualization
}

@tracer.chain()
def handle_tool_calls(tool_calls, messages):
    parts = []
    for tool_call in tool_calls:   
        function = tool_implementations[tool_call.name]
        function_args = tool_call.args
        result = function(**function_args)
        parts.append(
            types.Part.from_function_response(
                name=tool_call.name,
                response={"result": result}
            )
        )
        
    messages.append(types.Content(role="user", parts=parts))
    return messages

SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the Store Sales Price Elasticity Promotions dataset.
CRITICAL INSTRUCTION: You possess NO internal knowledge of the sales data. You MUST NEVER attempt to answer questions about store performance, sales, or trends without FIRST calling the `lookup_sales_data` tool. 
If a user asks about sales data, you MUST immediately output a tool call to `lookup_sales_data`. Do NOT hallucinate or make up store names or sales figures under any circumstances.
Step 1: Always use the `lookup_sales_data` tool to retrieve the relevant dataset.
Step 2: Only after receiving the tool's output, use the `analyze_sales_data` tool if insights are requested, or the `generate_visualization` tool if a chart is requested.
"""

def convert_to_openai_format(chat_messages):
    formatted = []
    for msg in chat_messages:
        if isinstance(msg, dict):
            formatted.append(msg)
            continue
        role = getattr(msg, "role", "unknown")
        for part in getattr(msg, "parts", []):
            if part.function_call:
                formatted.append({
                    "role": role,
                    "function_call": {
                        "name": part.function_call.name,
                        "arguments": str(part.function_call.args)
                    }
                })
            elif hasattr(part, "function_response") and part.function_response:
                formatted.append({
                    "role": role,
                    "name": part.function_response.name,
                    "content": str(part.function_response.response)
                })
            elif part.text:
                formatted.append({
                    "role": role,
                    "content": part.text
                })
    return formatted

def run_agent(messages):
    print("Running agent with messages:", messages)
    if isinstance(messages, str):
        chat_messages = [types.Content(role="user", parts=[types.Part.from_text(text=messages)])]
    else:
        chat_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                chat_messages.append(types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["content"])]))
            else:
                chat_messages.append(msg)

    while True:
        print("Starting router call span")
        with tracer.start_as_current_span(
            "router_call", openinference_span_kind="chain",
        ) as span:
            span.set_input(value=str(chat_messages))
            
            with tracer.start_as_current_span("router_llm", openinference_span_kind="llm") as llm_span:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=chat_messages,
                    config=types.GenerateContentConfig(
                        tools=tools_list,
                        system_instruction=SYSTEM_PROMPT,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                llm_span.set_output(value=response.text if response.text else str(response.function_calls))
                llm_span.set_status(StatusCode.OK)
            
            if response.candidates and response.candidates[0].content:
                chat_messages.append(response.candidates[0].content)
                
            tool_calls = response.function_calls
            span.set_status(StatusCode.OK)
    
            if tool_calls:
                print("Starting tool calls span")
                chat_messages = handle_tool_calls(tool_calls, chat_messages)
                span.set_output(value=str(tool_calls))
            else:
                print("No tool calls, returning final response")
                span.set_output(value=response.text)
                return convert_to_openai_format(chat_messages)

def start_main_span(messages):
    print("Starting main span with messages:", messages)
    
    with tracer.start_as_current_span(
        "AgentRun", openinference_span_kind="agent"
    ) as span:
        span.set_input(value=str(messages))
        ret = run_agent(messages)
        print("Main span completed with return value:", ret)
        span.set_output(value=ret)
        span.set_status(StatusCode.OK)
        return ret
