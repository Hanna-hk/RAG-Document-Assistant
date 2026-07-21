from unittest.mock import MagicMock
import ingest
import numpy as np

def test_process_pdf_calls_insert(mocker, tmp_path):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Some test content here."

    mock_reader = MagicMock()
    mock_reader.pages =[mock_page]

    mocker.patch("ingest.PdfReader", return_value=mock_reader)
    mocker.patch.object(ingest.model, "encode", return_value=np.array([[0.1, 0.2, 0.3]]))

    mock_table = MagicMock()
    mocker.patch.object(ingest.supabase, "table", return_value=mock_table)


    mock_table.insert.return_value.execute.return_value = MagicMock()

    ingest.process_pdf("fake.pdf")
    mock_table.insert.assert_called_once()