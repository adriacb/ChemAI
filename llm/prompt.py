from pydantic import BaseModel
# https://github.com/MTxSouza/MediumArticleGenerator/blob/main/app/utils.py



def check_prompt_structure(prompt: PromptStructure) -> PromptStructure:
    """Check the structure of the prompt."""
    raise NotImplementedError

from titan.llm.tools import PromptStructure

class PromptStructure(BaseModel):
    text: str
    max_tokens: int = 100