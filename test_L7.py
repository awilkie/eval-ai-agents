import pytest
import sys
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock dependencies before importing L7 to prevent network execution on import
mock_client_patch = patch('phoenix.client.Client')
mock_client = mock_client_patch.start()
mock_instance = MagicMock()
# Mock get_spans_dataframe to return an empty DataFrame with necessary columns to skip evaluation blocks safely
mock_instance.spans.get_spans_dataframe.return_value = pd.DataFrame(columns=["tool_call", "question", "generated_code"])
mock_client.return_value = mock_instance

mock_start_patch = patch('utils.start_main_span')
mock_start_patch.start()

import L7

def test_code_is_runnable_success():
    code = """
x = 5
y = 10
z = x + y
    """
    assert L7.code_is_runnable(code) == True

def test_code_is_runnable_failure():
    code = """
x = 5
y = 
    """
    assert L7.code_is_runnable(code) == False

def test_code_is_runnable_strips_markdown():
    code = """```python
x = 5
y = 10
```"""
    assert L7.code_is_runnable(code) == True

def test_clarity_prompt_defined():
    assert "clear" in L7.CLARITY_LLM_JUDGE_PROMPT
    assert "unclear" in L7.CLARITY_LLM_JUDGE_PROMPT

def test_sql_eval_prompt_defined():
    assert "correct" in L7.SQL_EVAL_GEN_PROMPT
    assert "incorrect" in L7.SQL_EVAL_GEN_PROMPT
