from pprint import pprint

from bgnlp import commatize


text = "За човек, който си ляга в 10 вечерта накъде съм ги изгубил наистина"

print("Without metadata:")
print(commatize(text))

print("\nWith metadata:")
pprint(commatize(text, return_metadata=True))
# Which prints:
# Without metadata:
# За човек, който си ляга в 10 вечерта, накъде съм ги изгубил наистина

# With metadata:
# ('За човек, който си ляга в 10 вечерта, накъде съм ги изгубил наистина',
#  [{'end': 14,
#    'score': 0.6685729026794434,
#    'start': 2,
#    'substring': ' човек, койт'},
#   {'end': 43,
#    'score': 0.9136603474617004,
#    'start': 28,
#    'substring': ' вечерта, накъд'}])
