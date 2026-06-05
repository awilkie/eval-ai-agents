#!/usr/bin/env python
# coding: utf-8

# # Lab 5: Adding Structure to your Evaluations 

# In this lab, you will run experiments:
# 
# <img src="images/experiments.png" width="500"/>

# to evaluate your agent example. Here are the evaluators you will use:
# 
# <img src="images/agent_evaluators.png" width="500"/>

# ## Importing necessary libraries 

import warnings
warnings.filterwarnings('ignore')

import phoenix as px
from phoenix.client.experiments import run_experiment, evaluate_experiment, create_evaluator
from phoenix.otel import register
import pandas as pd
from utils import get_phoenix_endpoint, run_agent, tool_schemas
from phoenix.client import Client as PhoenixClient

from datetime import datetime
import json
import os
from google import genai
import utils

# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code>, <code>utils.py</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

# ## Creating the Dataset of Test Questions

eval_model = "gemini-2.5-flash"
try:
    client = genai.Client(api_key=utils.gemini_api_key)
except Exception:
    client = None

px_client = PhoenixClient()

overall_experiment_questions = [
    {'question': 'What was the most popular product SKU?',
     'sql_result': '   SKU_Coded  Total_Qty_Sold 0    6200700         52262.0', 
     'sql_generated': '```sql\nSELECT SKU_Coded, SUM(Qty_Sold) AS Total_Qty_Sold\nFROM sales\nGROUP BY SKU_Coded\nORDER BY Total_Qty_Sold DESC\nLIMIT 1;\n```'},
    {'question': 'What was the total revenue across all stores?', 
     'sql_result': '   Total_Revenue 0   1.327264e+07', 
     'sql_generated': '```sql\nSELECT SUM(Total_Sale_Value) AS Total_Revenue\nFROM sales;\n```'},
    {'question': 'Which store had the highest sales volume?',
     'sql_result': '   Store_Number  Total_Sales_Volume 0          2970             59322.0', 
     'sql_generated': '```sql\nSELECT Store_Number, SUM(Total_Sale_Value) AS Total_Sales_Volume\nFROM sales\nGROUP BY Store_Number\nORDER BY Total_Sales_Volume DESC\nLIMIT 1;\n```'},
    {'question': 'Create a bar chart showing total sales by store',
     'sql_result': '    Store_Number    Total_Sales 0            880  420302.088397 1           1650  580443.007953 2           4180  272208.118542 3            550  229727.498752 4           1100  497509.528013 5           3300  619660.167018 6           3190  335035.018792 7           2970  836341.327191 8           3740  359729.808228 9           2530  324046.518720 10          4400   95745.620250 11          1210  508393.767785 12           330  370503.687331 13          2750  453664.808068 14          1980  242290.828499 15          1760  350747.617798 16          3410  410567.848126 17           990  378433.018639 18          4730  239711.708869 19          4070  322307.968330 20          3080  495458.238811 21          2090  309996.247965 22          1320  592832.067579 23          2640  308990.318559 24          1540  427777.427815 25          4840  389056.668316 26          2860  132320.519487 27          2420  406715.767402 28           770  292968.918642 29          3520  145701.079372 30           660  343594.978075 31          3630  405034.547846 32          2310  412579.388504 33          2200  361173.288199 34          1870  401070.997685', 
     'sql_generated': '```sql\nSELECT Store_Number, SUM(Total_Sale_Value) AS Total_Sales\nFROM sales\nGROUP BY Store_Number;\n```'},
    {'question': 'What percentage of items were sold on promotion?',
     'sql_result': '   Promotion_Percentage 0              0.625596',
     'sql_generated': "```sql\nSELECT \n    (SUM(CASE WHEN On_Promo = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS Promotion_Percentage\nFROM \n    sales;\n```"},
    {'question': 'What was the average transaction value?',
     'sql_result': '   Average_Transaction_Value 0                  19.018132',
     'sql_generated': '```sql\nSELECT AVG(Total_Sale_Value) AS Average_Transaction_Value\nFROM sales;\n```'},
    {'question': 'Create a line chart showing sales in 2021',
     'sql_result': '  sale_month  total_quantity_sold  total_sales_value 0 2021-11-01              43056.0      499984.428193 1 2021-12-01              75724.0      910982.118423', 
     'sql_generated': '```sql\nSELECT MONTH(Sold_Date) AS Month, SUM(Total_Sale_Value) AS Total_Sales\nFROM sales\nWHERE YEAR(Sold_Date) = 2021\nGROUP BY MONTH(Sold_Date)\nORDER BY MONTH(Sold_Date);\n```'}
]

overall_experiment_df = pd.DataFrame(overall_experiment_questions)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# create a dataset consisting of input questions and expected outputs
dataset = px_client.datasets.create_dataset(dataframe=overall_experiment_df, 
                                   name=f"overall_experiment_inputs-{now}", 
                                   input_keys=["question"], 
                                   output_keys=["sql_result", "sql_generated"])


# ### Link to Phoenix UI

# You can open this link to check out the Phoenix UI and the uploaded dataset. You can use the same link to check out the results of the experiment you'll run in this notebook. 
# 
# **Note**: 
# - Since each notebook of this course runs in an isolated environment, each notebook links to a different Phoenix server. This is why you won't see the projects you've worked on in the previous notebooks. 
# - Make sure that the notebook's kernel is running when checking the Phoenix UI. If the link does not open, it might be because the notebook has been open or inactive for a long time. In that case, make sure to refresh the browser, run all previous cells and then check this link. 

print(get_phoenix_endpoint())

# ## Setting up the Evaluators

# Let's define the template prompts for the LLM-as-a-judge that will be used to evaluate the clarity of the analysis of tool 2 and the correctness of the entities mentioned in the analysis. For router evals, you will use the template provided by Phoenix.

CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to evaluate the clarity 
of the answer in addressing the query. A clear response is one that is precise, coherent, and directly 
addresses the query without introducing unnecessary complexity or ambiguity. An unclear response is one 
that is vague, disorganized, or difficult to understand, even if it may be factually correct.

Your response should be a single word: either "clear" or "unclear," and it should not include any other 
text or characters. "clear" indicates that the answer is well-structured, easy to understand, and 
appropriately addresses the query. "unclear" indicates that the answer is ambiguous, poorly organized, or 
not effectively communicated. Please carefully consider the query and answer before determining your 
response.

After analyzing the query and the answer, you must write a detailed explanation of your reasoning to 
justify why you chose either "clear" or "unclear." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the answer does or does not meet the 
criteria for clarity.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the clarity of the answer based on the query.
LABEL: "clear" or "unclear"
"""

ENTITY_CORRECTNESS_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to determine whether all 
the entities mentioned in the answer are correctly identified and accurately match those in the query. An 
entity refers to any specific person, place, organization, date, or other proper noun. Your evaluation 
should focus on whether the entities in the answer are correctly named and appropriately associated with 
the context in the query.

Your response should be a single word: either "correct" or "incorrect," and it should not include any 
other text or characters. "correct" indicates that all entities mentioned in the answer match those in the 
query and are properly identified. "incorrect" indicates that the answer contains errors or mismatches in 
the entities referenced compared to the query.

After analyzing the query and the answer, you must write a detailed explanation of your reasoning to 
justify why you chose either "correct" or "incorrect." Avoid stating the final label at the beginning of 
your explanation. Your reasoning should include specific points about how the entities in the answer do or 
do not match the entities in the query.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating whether the entities in the answer are 
correct and consistent with the query.
LABEL: "correct" or "incorrect"
"""

TOOL_CALLING_PROMPT_TEMPLATE = """
You are evaluating the accuracy of a tool call made by an AI agent.
Below is the set of available tools, the user's question, and the tool call made by the agent.

[AVAILABLE TOOLS]
{tool_definitions}

[USER QUESTION]
{question}

[AGENT TOOL CALL]
{tool_call}

Evaluate whether the agent made the correct tool call and provided the correct parameters to satisfy the user's question.
Your response must be a single word: either "correct" or "incorrect".
If the tool call matches the best tool for the question and the arguments are appropriate, respond with "correct".
If the agent chose the wrong tool, hallucinated a tool, or provided bad arguments, respond with "incorrect".
"""

# The following evaluators are set up to take in as parameters: input, output and expected. Here are the structures and meaning of these parameters:
# - **input**: this is the input field of your dataset examples that you just created. It has only one key: "question" (as defined in a previous cell).
# - **output**: this is the output field added to your dataset examples, after you apply the task to each example. The structure of this output is defined by the task, which is defined in a subsequent cell (as `run_agent_task`). This task returns a processed version of the agent's messages (you can check `process_messages` in `utils.py`): it is a dictionary that organizes the messages into these keys: "tool_calls", "tool_responses", "final_output", "unchanged_messages" and "path_length".
# - **expected**: this is the expected output field of your dataset examples that you created in a previous cell. It has two keys: "sql_result" and "sql_generated".

@create_evaluator(name="Function Calling", kind="CODE")
def function_calling_eval(output: dict, input: dict) -> float:
    if output is None:
        return 0.0
    function_calls = output.get("tool_calls")
    if not function_calls:
        return 0.0
    
    scores = []
    for tc in function_calls:
        prompt = TOOL_CALLING_PROMPT_TEMPLATE.format(
            tool_definitions=json.dumps(tool_schemas).replace("{", '"').replace("}", '"'),
            question=input.get("question"),
            tool_call=str(tc)
        )
        try:
            response = client.models.generate_content(
                model=eval_model,
                contents=prompt,
            )
            text = response.text.lower()
            if "incorrect" in text:
                scores.append(0.0)
            elif "correct" in text:
                scores.append(1.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)
    
    return sum(scores)/len(scores) if scores else 0.0

@create_evaluator(name="SQL Result", kind="CODE")
def evaluate_sql_result(output: dict, expected: dict) -> float:    
    if output is None:
        return 0.0
    sql_result = output.get("tool_responses")
    if not sql_result:
        return 1.0
    
    # Find first lookup_sales_data response
    sql_result = next((r for r in sql_result if r.get("tool_name") == "lookup_sales_data"), None)
    if not sql_result:
        return 1.0
        
    # Get the first response
    sql_result = sql_result.get("tool_response", "")

    # Extract just the numbers from both strings
    result_nums = ''.join(filter(str.isdigit, sql_result))
    expected_nums = ''.join(filter(str.isdigit, expected.get("sql_result", "")))
    return 1.0 if result_nums == expected_nums else 0.0

@create_evaluator(name="Clarity", kind="CODE")
def evaluate_clarity(output: dict, input: dict) -> float:
    if output is None or not output.get("final_output"):
        return 0.0
    prompt = CLARITY_LLM_JUDGE_PROMPT.format(
        query=input.get("question"),
        response=output.get("final_output")
    )
    try:
        response = client.models.generate_content(model=eval_model, contents=prompt)
        if "unclear" in response.text.lower():
            return 0.0
        elif "clear" in response.text.lower():
            return 1.0
        return 0.0
    except Exception:
        return 0.0

@create_evaluator(name="Entity Correctness", kind="CODE")
def evaluate_entity_correctness(output: dict, input: dict) -> float:
    if output is None or not output.get("final_output"):
        return 0.0
    prompt = ENTITY_CORRECTNESS_LLM_JUDGE_PROMPT.format(
        query=input.get("question"),
        response=output.get("final_output")
    )
    try:
        response = client.models.generate_content(model=eval_model, contents=prompt)
        if "incorrect" in response.text.lower():
            return 0.0
        elif "correct" in response.text.lower():
            return 1.0
        return 0.0
    except Exception:
        return 0.0

@create_evaluator(name="Code Runnable", kind="CODE")
def code_is_runnable(output: dict) -> float:
    """Check if the code is runnable"""
    if output is None:
        return 0.0
    generated_code = output.get("tool_responses")
    if not generated_code:
        return 1.0
    
    # Find first generate_visualization response
    generated_code = next((r for r in generated_code if r.get("tool_name") == "generate_visualization"), None)
    if not generated_code:
        return 1.0
        
    # Get the first response
    generated_code = generated_code.get("tool_response", "")
    generated_code = generated_code.strip()
    generated_code = generated_code.replace("```python", "").replace("```", "")
    try:
        exec(generated_code)
        return 1.0
    except Exception as e:
        return 0.0


# ## Defining the Task

def process_messages(messages):
    tool_calls = []
    tool_responses = []
    final_output = ""
    for msg in messages:
        if "function_call" in msg:
            import ast
            try:
                args = ast.literal_eval(msg["function_call"]["arguments"])
            except Exception:
                args = msg["function_call"]["arguments"]
            tool_calls.append({
                "name": msg["function_call"]["name"],
                "args": args
            })
        elif "name" in msg and "content" in msg:
            tool_responses.append({
                "tool_name": msg["name"],
                "tool_response": msg["content"]
            })
        elif "content" in msg and msg.get("role") in ["model", "assistant"]:
            final_output = msg["content"]
            
    return {
        "tool_calls": tool_calls,
        "tool_responses": tool_responses,
        "final_output": final_output,
        "unchanged_messages": messages,
        "path_length": len(messages)
    }

def run_agent_task(example) -> dict:
    print("Starting agent with messages:", example.input.get("question"))
    messages = [{"role": "user", "content": example.input.get("question")}]
    ret = run_agent(messages)
    return process_messages(ret)

def update_sql_gen_prompt(new_prompt):
    import utils
    utils.SQL_GENERATION_PROMPT = new_prompt

def export_overall_evaluations_to_ascii(experiment1, experiment2):
    try:
        import pandas as pd
        from datetime import datetime
        print("Exporting overall evaluations to ASCII visualization...")
        
        def get_avg_scores(exp):
            scores = {}
            for eval_run in exp.get("evaluation_runs", []):
                name = getattr(eval_run, "name", "")
                res = getattr(eval_run, "result", None)
                if isinstance(res, dict) and "score" in res:
                    scores.setdefault(name, []).append(res["score"])
                elif isinstance(res, (int, float)):
                    scores.setdefault(name, []).append(res)
            
            avg_scores = {}
            for name, sc_list in scores.items():
                avg_scores[name] = sum(sc_list) / len(sc_list) if sc_list else 0.0
            return avg_scores
            
        avg1 = get_avg_scores(experiment1)
        avg2 = get_avg_scores(experiment2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"overall_evaluations_ascii_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write("========================================================================\n")
            f.write("                 OVERALL EXPERIMENT RESULTS VISUALISATION               \n")
            f.write("========================================================================\n\n")
            f.write("Experiment 1: Overall Experiment\n")
            f.write("-" * 72 + "\n")
            for name, score in avg1.items():
                bar = "█" * int(score * 40)
                f.write(f"  {name[:20]:<20} | {bar:<40} | {score*100:.1f}%\n")
            
            f.write("\n")
            f.write("Experiment 2: Overall Experiment v2 (Updated SQL Prompt)\n")
            f.write("-" * 72 + "\n")
            for name, score in avg2.items():
                bar = "█" * int(score * 40)
                f.write(f"  {name[:20]:<20} | {bar:<40} | {score*100:.1f}%\n")
                
            f.write("\n========================================================================\n")
            
        print(f"Overall evaluations ASCII visualization exported to {filename}")
    except Exception as e:
        print(f"Failed to generate ASCII visualization: {e}")

# ## Running the Experiment

# <p style="background-color:#f7fff8; padding:15px; border-width:3px; border-color:#e0f0e0; border-style:solid; border-radius:6px"> 🚨
# &nbsp; <b>Different Run Results:</b> The output generated by AI chat models can vary with each execution due to their dynamic, probabilistic nature. Your results might differ from those shown in the video. The warnings are also not printed in this notebook.</p> 

if __name__ == "__main__":
    print("Running experiment 1")
    experiment1 = run_experiment(dataset=dataset,
                                task=run_agent_task,
                                experiment_name="Overall Experiment",
                                experiment_description="Evaluating the overall experiment")

    experiment1 = evaluate_experiment(experiment=experiment1,
                                evaluators=[function_calling_eval,
                                            evaluate_sql_result, 
                                            evaluate_clarity, 
                                            evaluate_entity_correctness, 
                                            code_is_runnable])

    # ## Running the Experiment - Change in Prompt
    new_prompt = """
    Generate an SQL query based on a prompt. 
    Do not reply with anything besides the SQL query.
    The prompt is: {prompt}

    The available columns are: {columns}
    The table name is: {table_name}

    Think before you respond.
    """

    update_sql_gen_prompt(new_prompt)

    print("Running experiment 2")
    experiment2 = run_experiment(dataset=dataset,
                                task=run_agent_task,
                                experiment_name="Overall Experiment v2",
                                experiment_description="Evaluating the overall experiment, with changes to sql prompt")

    experiment2 = evaluate_experiment(experiment=experiment2,
                                evaluators=[function_calling_eval, 
                                            evaluate_sql_result, 
                                            evaluate_clarity, 
                                            evaluate_entity_correctness,
                                            code_is_runnable])

    # ## Optional - Playground in Phoenix
    import utils
    print(utils.SQL_GENERATION_PROMPT)

    export_overall_evaluations_to_ascii(experiment1, experiment2)
