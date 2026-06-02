import pytest
from unittest.mock import patch, MagicMock
from google.genai import types
import L3

def test_lookup_sales_data(monkeypatch):
    # Mock DuckDB to avoid needing the actual parquet file
    mock_df = MagicMock()
    mock_df.to_string.return_value = "store_id | sales\n1320 | 500"
    
    mock_duckdb = MagicMock()
    mock_duckdb.df.return_value = mock_df
    
    # We patch duckdb.sql to return our mock duckdb result
    def mock_duckdb_sql(query):
        return mock_duckdb
        
    monkeypatch.setattr(L3.duckdb, "sql", mock_duckdb_sql)
    
    # We also need to mock generate_sql_query so it doesn't make an API call
    monkeypatch.setattr(L3, "generate_sql_query", lambda prompt, cols, table: "SELECT * FROM sales")
    
    # And mock pd.read_parquet
    monkeypatch.setattr(L3.pd, "read_parquet", lambda path: MagicMock(columns=["store_id", "sales"]))
    
    result = L3.lookup_sales_data("Show me all the sales for store 1320")
    assert "store_id | sales" in result
    assert "1320 | 500" in result

@patch('L3.client.models.generate_content')
def test_analyze_sales_data(mock_generate_content):
    mock_response = MagicMock()
    mock_response.text = "Sales are increasing."
    mock_generate_content.return_value = mock_response
    
    result = L3.analyze_sales_data("What trends do you see?", "store_id | sales\n1320 | 500")
    assert result == "Sales are increasing."
    assert mock_generate_content.called

@patch('L3.client.models.generate_content')
def test_extract_chart_config(mock_generate_content):
    mock_response = MagicMock()
    mock_response.text = '{"chart_type": "bar", "x_axis": "store_id", "y_axis": "sales", "title": "Sales by Store"}'
    mock_generate_content.return_value = mock_response
    
    config = L3.extract_chart_config("data", "Sales by Store")
    assert config["chart_type"] == "bar"
    assert config["title"] == "Sales by Store"

@patch('L3.client.models.generate_content')
def test_create_chart(mock_generate_content):
    mock_response = MagicMock()
    mock_response.text = "```python\nimport matplotlib.pyplot as plt\n```"
    mock_generate_content.return_value = mock_response
    
    code = L3.create_chart({"chart_type": "bar"})
    assert "import matplotlib.pyplot as plt" in code
    assert "```" not in code

@patch('L3.client.models.generate_content')
def test_run_agent_flow(mock_generate_content):
    # Setup mock to return a text response directly (no tool calls)
    mock_response = MagicMock()
    mock_response.function_calls = None
    mock_response.text = "Final answer"
    mock_candidate = MagicMock()
    mock_candidate.content = MagicMock()
    mock_response.candidates = [mock_candidate]
    
    mock_generate_content.return_value = mock_response
    
    result = L3.run_agent("Hello")
    assert result == "Final answer"

@patch('L3.create_chart')
@patch('L3.extract_chart_config')
def test_generate_visualization(mock_extract, mock_create):
    mock_extract.return_value = {"chart_type": "bar"}
    mock_create.return_value = "print('dummy code executed')"
    
    result = L3.generate_visualization("data", "goal")
    assert "Chart successfully generated" in result
    assert "print('dummy code executed')" in result

def test_system_prompt_instructions():
    assert "CRITICAL INSTRUCTION:" in L3.SYSTEM_PROMPT
    assert "generate_visualization" in L3.SYSTEM_PROMPT

