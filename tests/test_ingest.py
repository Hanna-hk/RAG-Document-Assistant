from unittest.mock import MagicMock
import ingest
import numpy as np

def test_process_pdf_calls_insert(mocker, tmp_path):
    """
        Verify that `process_pdf` inserts data into Supabase after processing a PDF.

        Mocks out all external dependencies (PdfReader, the embedding model,
        and the Supabase client) so the test runs in isolation without needing
        a real PDF file, model weights, or network access. Asserts that the
        Supabase "insert" call is made exactly once, confirming that the
        ingestion pipeline (read -> chunk -> embed -> store) completes and
        reaches the database write step.

        Args:
            mocker: pytest-mock fixture used to patch dependencies.
            tmp_path: pytest built-in fixture providing a temporary directory.
    """
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