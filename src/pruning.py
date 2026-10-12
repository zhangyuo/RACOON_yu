# RAG4CTA fix: this module used to talk to the real OpenAI API via
# langchain_community's deprecated ChatOpenAI, independently of
# utils.response()'s model swap. Pointed at our vLLM/GLM server, BUT that old
# langchain client silently drops GLM/vLLM's `reasoning` field (where this
# model actually puts its answer -- see utils.response()'s fix for the main
# generation calls) and exposes neither `.content` nor it in
# `additional_kwargs`/`response_metadata` -- confirmed empty even when the
# server reports real completion_tokens generated. Every pruning_orig/
# pruning_rel call was silently returning "" for every table in every run
# (SC, DB, and TURL alike), wiping out the KG hint regardless of whether
# entity linking itself found anything. Reusing utils.response() (the same
# already-patched OpenAI client) instead of a second, broken client fixes
# this without a new fallback to maintain.
from utils import response as _llm_response


def _ask(system_prompt, user_prompt):
    msg = _llm_response(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=None,
    )
    return (msg.content or "").strip()

def pruning_rel(table, hint):
    """
    Use an LLM to prune the retrieved entity relations from the KG.
    Returns: pruned hint.
    """
    system_prompt = """
        You are a careful and precise assistant. Your task is to prune a hint block associated with tabular data.
        ONLY remove parts of the original hint that are clearly incorrect, irrelevant, redundant, or do not align with the actual column values.

        DO NOT add new interpretations, rephrase, or insert descriptions that were not in the original hint.
        DO NOT change the structure or wording of any valid content.
        DO NOT add new types or labels.

        Keep the output format exactly the same:
        - Column 1 & Column N: <original hint with only clearly unhelpful or inaccurate parts removed>

        Include each column pair in your output even if the pruned hint is empty for that column pair. Your output should be a string, not a list.

        If a column's hint is already empty, keep it that way.
        Return only the pruned hint block.
                """
    user_prompt = f"""
        Table:
        {table}

        Original hint:
        {hint}
                """
    return _ask(system_prompt, user_prompt)

def pruning_orig(table, hint):
    """
    Use an LLM to prune the retrieved hint (entity labels, entity labels + descriptions, or entity types) from the KG.
    Returns: pruned hint.
    """
    system_prompt = """
        You are a careful and precise assistant. Your task is to prune a hint block associated with tabular data.
        ONLY remove parts of the original hint that are clearly incorrect, irrelevant, redundant, or do not align with the actual column values.

        DO NOT add new interpretations, rephrase, or insert descriptions that were not in the original hint.
        DO NOT change the structure or wording of any valid content.
        DO NOT add new types or labels.

        Keep the output format exactly the same:
        - Column N: <original hint with only clearly unhelpful or inaccurate parts removed>

        Your output should be a string, not a list. Your output should include all the columns in the original hint even if the pruned hint is empty for some columns.

        If a column's hint is already empty, keep it that way.
        Return only the pruned hint block.
                """
    user_prompt = f"""
        Table:
        {table}

        Original hint:
        {hint}
                """
    return _ask(system_prompt, user_prompt)


