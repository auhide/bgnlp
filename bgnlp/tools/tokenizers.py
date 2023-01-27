import re

from bgnlp.lib.utils import create_actions_pipeline


# TODO: Implement basic functionality for inheritance.
class Tokenizer:
    pass


class DefaultPreTokenizer(Tokenizer):
    # These actions are going to be executed sequentially, based on the order 
    # in which they are implemented in the class.
    actions_list = []
    action = create_actions_pipeline(actions_list)

    def __call__(self, string, split_type="word"):
        super().__init__()

        self.string = string
        
        for action_func in self.actions_list:
            self.string = action_func(self)

        if split_type == "word":
            return self.string.split(" ")
        elif split_type == "symbol":
            return [symbol for symbol in self.string]

    @action
    def pad_punctuation(self):
        return re.sub(r"([!,\.\"\'\?\-])", r" \1 ", self.string)

    @action
    def remove_multiple_whitespaces(self):
        return re.sub(r"\s+", " ", self.string)

    @action
    def remove_outer_whitespace(self):
        return self.string.strip()


# tokenizer = DefaultPreTokenizer()
# tokens = tokenizer("Не мога да живея повече!!! Това не може да бъде по-трудно.")
# print(tokens)
