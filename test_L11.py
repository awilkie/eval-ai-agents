import pytest
from unittest.mock import patch, MagicMock
import L11

def test_evaluate_sql_result():
    output_correct = {"tool_responses": [{"tool_name": "lookup_sales_data", "tool_response": "Result is 123"}]}
    expected = {"sql_result": "Value: 123"}
    assert L11.evaluate_sql_result(output_correct, expected) == 1.0
    
    output_incorrect = {"tool_responses": [{"tool_name": "lookup_sales_data", "tool_response": "Result is 456"}]}
    assert L11.evaluate_sql_result(output_incorrect, expected) == 0.0
    
    # Missing lookup_sales_data
    assert L11.evaluate_sql_result({"tool_responses": []}, expected) == 1.0

def test_code_is_runnable():
    output_runnable = {"tool_responses": [{"tool_name": "generate_visualization", "tool_response": "a = 1 + 1"}]}
    assert L11.code_is_runnable(output_runnable) == 1.0
    
    output_not_runnable = {"tool_responses": [{"tool_name": "generate_visualization", "tool_response": "a = 1 + /"}]}
    assert L11.code_is_runnable(output_not_runnable) == 0.0

@patch('L11.client')
def test_evaluate_clarity(mock_client):
    mock_response = MagicMock()
    mock_response.text = "clear"
    mock_client.models.generate_content.return_value = mock_response
    
    output = {"final_output": "The sales were high."}
    input_data = {"question": "How were the sales?"}
    assert L11.evaluate_clarity(output, input_data) == 1.0

@patch('L11.client')
def test_evaluate_entity_correctness(mock_client):
    mock_response = MagicMock()
    mock_response.text = "incorrect"
    mock_client.models.generate_content.return_value = mock_response
    
    output = {"final_output": "The sales for Store 2 were high."}
    input_data = {"question": "How were the sales for Store A?"}
    assert L11.evaluate_entity_correctness(output, input_data) == 0.0

@patch('L11.client')
def test_function_calling_eval(mock_client):
    mock_response = MagicMock()
    mock_response.text = "correct"
    mock_client.models.generate_content.return_value = mock_response
    
    output = {"tool_calls": [{"name": "lookup_sales_data", "args": {"prompt": "sales"}}]}
    input_data = {"question": "What were the sales?"}
    assert L11.function_calling_eval(output, input_data) == 1.0
