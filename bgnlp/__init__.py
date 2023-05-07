from bgnlp.tools.taggers import (
    PosTagger, 
    LemmaTagger, 
    NerTagger
)
from bgnlp.tools.configs import (
    PosTaggerConfig, 
    LemmaTaggerConfig, 
    NerTaggerConfig
)


pos = PosTagger(config=PosTaggerConfig())
lemmatize = LemmaTagger(config=LemmaTaggerConfig())
ner = NerTagger(config=NerTaggerConfig())


__version__ = "0.2.0"
