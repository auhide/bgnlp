from bgnlp.tools.taggers import (
    PosTagger, 
    LemmaTagger, 
    NerTagger,
    KeywordsTagger,
    PunctuationTagger,
)
from bgnlp.tools.configs import (
    PosTaggerConfig, 
    LemmaTaggerConfig, 
    NerTaggerConfig,
    KeywordsTaggerConfig,
    PunctuationTaggerConfig,
)


pos = PosTagger(config=PosTaggerConfig())
lemmatize = LemmaTagger(config=LemmaTaggerConfig())
ner = NerTagger(config=NerTaggerConfig())
extract_keywords = KeywordsTagger(config=KeywordsTaggerConfig())
commatize = PunctuationTagger(config=PunctuationTaggerConfig())


__version__ = "0.5.2"
