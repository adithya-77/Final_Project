import re
import logging

class TextChunker:
    def __init__(self, max_chars=4000):
        """
        Initialize the text chunker.
        
        Args:
            max_chars: Maximum characters per chunk (approximation for tokens)
        """
        self.logger = logging.getLogger(__name__)
        self.max_chars = max_chars
    
    def chunk_text(self, text):
        """
        Split text into chunks of appropriate size for the LLaMA model.
        Tries to split at paragraph boundaries when possible.
        """
        if not text.strip():
            return []
        
        # Split by paragraphs (double newlines or periods followed by whitespace)
        paragraphs = re.split(r'\n\s*\n|\.\s+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            # If paragraph is too long by itself, split it by sentences
            if len(paragraph) > self.max_chars:
                # Process the current chunk if it's not empty
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                for sentence in sentences:
                    if len(sentence) > self.max_chars:
                        # For extremely long sentences, split by character count
                        for i in range(0, len(sentence), self.max_chars):
                            chunks.append(sentence[i:i + self.max_chars])
                    elif current_length + len(sentence) <= self.max_chars:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                    else:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
            
            # Normal paragraph handling
            elif current_length + len(paragraph) <= self.max_chars:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        self.logger.info(f"Split text into {len(chunks)} chunks of approximately {self.max_chars} characters each")
        return chunks
