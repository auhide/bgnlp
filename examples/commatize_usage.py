from pprint import pprint

from bgnlp import commatize


text = "Човекът искащ безгрижно писане ме помоли да създам този модел."

print("Without metadata:")
print(commatize(text))

print("\nWith metadata:")
pprint(commatize(text, return_metadata=True))
# Which prints:
# Without metadata:
# Човекът, искащ безгрижно писане, ме помоли да създам този модел.

# With metadata:
# ('Човекът, искащ безгрижно писане, ме помоли да създам този модел.',
#  [{'end': 12,
#    'score': 0.9301406145095825,
#    'start': 0,
#    'substring': 'Човекът, иск'},
#   {'end': 34,
#    'score': 0.93571537733078,
#    'start': 24,
#    'substring': ' писане, м'}])
