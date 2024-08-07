from pydantic import BaseModel
# https://github.com/MTxSouza/MediumArticleGenerator/blob/main/app/utils.py
class PromptStructure(BaseModel):
    text: str
    max_tokens: int = 100


def check_prompt_structure(prompt: PromptStructure) -> PromptStructure:
    """Check the structure of the prompt."""
    raise NotImplementedError