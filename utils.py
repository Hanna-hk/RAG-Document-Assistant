def chunk_text(text, chunk_size=500, overlap=50):
    """
        Split text into overlapping chunks of words.

        The text is split on whitespace into words, then grouped into chunks
        of `chunk_size` words each. Consecutive chunks share `overlap` words
        at their boundary, which helps preserve context that would otherwise
        be cut off at a hard split point.

        Args:
            text (str): The input text to split into chunks.
            chunk_size (int, optional): Number of words per chunk. Defaults to 500.
            overlap (int, optional): Number of words shared between consecutive
                chunks. Must be smaller than `chunk_size`.

        Returns:
            list[str]: A list of text chunks. Returns an empty list if `text`
                is empty.
        """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    words = text.split()
    chunks=[]

    for i in range (0, len(words), chunk_size-overlap):
        chunk = " ". join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks