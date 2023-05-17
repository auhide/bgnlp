from bgnlp.tools.taggers import (
    PosTagger, 
    LemmaTagger, 
    NerTagger,
    KeywordsTagger
)
from bgnlp.tools.configs import (
    PosTaggerConfig, 
    LemmaTaggerConfig, 
    NerTaggerConfig,
    KeywordsTaggerConfig
)


pos = PosTagger(config=PosTaggerConfig())
lemmatize = LemmaTagger(config=LemmaTaggerConfig())
ner = NerTagger(config=NerTaggerConfig())
extract_keywords = KeywordsTagger(config=KeywordsTaggerConfig())


__version__ = "0.3.1"
