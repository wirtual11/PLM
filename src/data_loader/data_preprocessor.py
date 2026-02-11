from __future__ import annotations
import re
import unicodedata

class Preprocessor:
    def __init__(self):
        pass

    def preprocess(self, data: dict[str, str]) ->list[str]:
        result: list[str] = []
        for fileContent in data.values():
            result.append(self.preprocess_single_text(fileContent))

        return result
    

    def preprocess_single_text(
            self,
            text: str,
            *,
            lowercase: bool = True,
            ascii_punct:bool = True,
            max_blank_lines:int = 2
          ) -> str:
        
         # 1) Unicode normalization
        text = unicodedata.normalize('NFKC', text)

         # 2) Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 3) Remove control chars except newline/tab
        # Keep chars where category doesn't start with 'C' (Other),
        # or keep \n/\t explicitly.
        text = "".join(ch for ch in text if ch in ('\n', '\t') or unicodedata.category(ch)[0] != 'C')

        # 4) Normalize whitespace
        # Tabs -> single space (or choose 4 spaces if you prefer)
        text = text.replace('\t', ' ')

        # Remove trailing spaces on each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        # Collapse multiple spaces within lines (but not newlines)
        text = re.sub(r"[ ]{2,}", " ", text)
        
        # 5) Limit blank lines
        # e.g. if max_blank_lines=2, allow at most two consecutive \n
        text = re.sub(r"\n{" + str(max_blank_lines + 1) + r",}", "\n" * max_blank_lines, text)

        # 6) Optional punctuation normalization
        if ascii_punct:
            replacements = {
                "\u2018": "'",  # left single quote
                "\u2019": "'",  # right single quote
                "\u201C": '"',  # left double quote
                "\u201D": '"',  # right double quote
                "\u2013": "-",  # en dash
                "\u2014": "-",  # em dash
                "\u2026": "...",# ellipsis
                "\u00A0": " ",  # non-breaking space
            }
            for src, dst in replacements.items():
                text = text.replace(src, dst)

        # 7) Optional lowercase
        if lowercase:
            text = text.lower() 

        return text