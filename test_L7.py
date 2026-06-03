import pytest
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
