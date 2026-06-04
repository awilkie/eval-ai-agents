import pytest
from unittest.mock import patch, MagicMock
import L9

def test_format_message_steps():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "System context"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "lookup_sales"}}]},
        {"role": "tool", "content": "sales data"}
    ]
    result = L9.format_message_steps(messages)
    assert "User: Hello" in result
    assert "System: Provided context" in result
    assert "Assistant: Hi there" in result
    assert "Assistant: Called tool 'lookup_sales'" in result
    assert "Tool response: sales data" in result

@patch('L9.run_agent')
def test_run_agent_and_track_path(mock_run_agent):
    mock_run_agent.return_value = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
    mock_example = MagicMock()
    mock_example.input = {"question": "What is the average transaction value?"}
    
    result = L9.run_agent_and_track_path(mock_example)
    
    assert result["path_length"] == 2
    assert "User: question" in result["messages"]
    assert "Assistant: answer" in result["messages"]

def test_evaluate_path_length(monkeypatch):
    monkeypatch.setattr(L9, 'optimal_path_length', 2)
    
    # When output has valid path_length
    assert L9.evaluate_path_length({"path_length": 4}) == 0.5
    assert L9.evaluate_path_length({"path_length": 2}) == 1.0
    
    # When output is empty or missing path_length
    assert L9.evaluate_path_length({}) == 0
    assert L9.evaluate_path_length({"path_length": 0}) == 0

def test_evaluate_path_length_wrong_type(monkeypatch):
    monkeypatch.setattr(L9, 'optimal_path_length', 2)
    # Check what happens when a non-dict is passed (as per our updated code logic)
    assert L9.evaluate_path_length("not a dict") == 0
