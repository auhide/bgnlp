from typing import List


class SubwordMixin:

    def subwords_to_words(self, tokens: List[str]) -> List[str]:
        """Covert subword tokens into word tokens.
        
        Example::
            >>> subword_tokens = ["▁Здра", "вей", "▁другарю", "!"]
            >>> tokens = SubwordMixin().subwords_to_words(subword_tokens)
            >>> tokens
            ["Здравей", "другарю", "!"]
        """
        out_tokens = []
        curr_token = ""
        
        for token in tokens:
            if token == "[SEP]":
                curr_token = curr_token.replace("▁", "")
                out_tokens.append(curr_token)
                out_tokens.append("[SEP]")
                break

            if "▁" in token and curr_token == "":
                curr_token += token

            elif "▁" in token and curr_token != "":
                curr_token = curr_token.replace("▁", "")
                out_tokens.append(curr_token)
                curr_token = ""
                curr_token += token

            elif "▁" not in token:
                curr_token += token

        return out_tokens