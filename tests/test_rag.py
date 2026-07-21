import pytest
from unittest.mock import MagicMock, patch

from proto.marshal.compat import message

import rag
import numpy as np

def test_search_context_returns_matches(mocker):
    mock_encode = mocker.patch.object(rag.model, "encode")
    mock_encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_response = MagicMock()
    mock_response.data = [
        {"id": 1, "content": "some text", "metadata": {"source": "test.pdf"}, "similarity": 0.8}
    ]
    mocker.patch.object(rag.supabase, "table").return_value.select.return_value.execute.return_value = MagicMock(count=10)
    mocker.patch.object(rag.supabase, "rpc").return_value.execute.return_value = mock_response
    result = rag.search_context("test query")
    assert len(result) == 1
    assert result[0]["metadata"]["source"] == "test.pdf"

def test_generate_answer_no_matches_returns_empty(mocker):
    mocker.patch.object(rag, "search_context", return_value=[])
    answer, sources = rag.generate_answer("random question")
    assert sources == []
    assert "no information" in answer.lower()

def test_generate_answer_with_matches_calls_llm(mocker):
    fake_matches = [
        {"content": "NovaLeaf was founded in 2018", "metadata": {"source": "example.pdf"}, "similarity": 0.9}
    ]
    mocker.patch.object(rag, "search_context", return_value = fake_matches)
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="NovaLeaf was founded in 2018."))]
    mocker.patch.object(rag.client.chat.completions, "create", return_value=mock_completion)
    answer, sources = rag.generate_answer("When was NovaLeaf founded?")
    assert "2018" in answer
    assert sources == fake_matches