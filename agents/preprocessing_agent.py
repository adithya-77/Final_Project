import re

class PreprocessingAgent:
    def clean(self, text):
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove non-alphanumeric characters (if necessary)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
