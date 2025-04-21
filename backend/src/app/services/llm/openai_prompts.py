"""The prompts for the language model."""

from string import Template

_CITATION_INSTRUCTIONS = """
**Output Format Instructions**:

- Respond with a **JSON object** containing two fields: "answer" and "cited_chunk_indices".
- The "answer" field should contain your answer to the **Question**, strictly following the specific format instructions provided below (e.g., boolean, string, integer array). Base the answer **only** on the provided **Context**. If the answer cannot be determined from the context, the "answer" field should be exactly `None`.
- The "cited_chunk_indices" field should be a JSON array of integers. Each integer must be the 0-based index of a chunk from the **Context** that was **essential** for formulating the answer (e.g., `[0, 2]`). The chunks in the Context are clearly marked like `--- Chunk 0 (Page: X) ---`.
- **Strict Citation Criteria**: Only include the index of a chunk if:
    - The chunk **directly contains the answer** or a significant part of the answer.
    - OR the chunk provides **essential context or information** without which the answer could not have been accurately determined or formulated based *solely* on the provided Context.
- Do **NOT** cite chunks that merely provide background information or were scanned but ultimately not used for the final answer.
- If no specific chunks meet these strict criteria (e.g., the answer is `None`, derived from general understanding across multiple chunks without specific reliance, or the context doesn't support an answer), provide an empty array `[]` or `null` for "cited_chunk_indices".
- Do not include any introductory or concluding remarks, markdown formatting, or explanations outside the JSON object.

Example JSON Response:
```json
{
  "answer": "Example answer based on format rules",
  "cited_chunk_indices": [1, 3]
}
```
"""

BASE_PROMPT = Template(
    f"""
You are an expert assistant whose job is to answer the following question using **only** the information provided in the **Context**. Do not use any prior knowledge or external information. Follow the output format instructions precisely.

---

**Question**: $query

---

**Context**:
$chunks

---

{_CITATION_INSTRUCTIONS}

---

**Specific Format Instructions for the "answer" field**:

$format_specific_instructions

---

**JSON Response**:
"""
)

INFERRED_BASE_PROMPT = Template(
    """
Answer the following question following the formatting instructions at the bottom. Do not include, quotes, formatting, or any explanation or extra information. Just answer the question.

**Question**: $query
**Answer**:

$format_specific_instructions

"""
)

BOOL_INSTRUCTIONS = """
**Special Instructions for Boolean Questions**:

- If the question is asking for a verification or requires a boolean answer, respond with True or False.
- If you cannot answer the question, respond exactly with 'None'.
- Do not provide any explanations or additional information.
"""

STR_ARRAY_INSTRUCTIONS = Template(
    """
$str_rule_line
$int_rule_line

**Special Instructions for String Responses**:

- If the answer is a single string, provide a single string.
- If multiple strings are required, provide them as a JSON array of strings.
- If you cannot find an answer, respond exactly with 'None'.
- Do not include any additional text or explanation.
"""
)

INT_ARRAY_INSTRUCTIONS = Template(
    """
$int_rule_line

**Special Instructions for Integer Responses**:

- If the answer is a single integer, provide the integer as a number.
- If multiple integers are required, provide them as a JSON array of integers.
- If you cannot find an answer, respond exactly with 'None'.
- Do not include any additional text or explanation.
"""
)

KEYWORD_PROMPT = Template(
    """
You are tasked with extracting the most relevant keywords from the following query. Focus on the main nouns and verbs that capture the essence of the query.

---

**Query**: $query

---

**Instructions**:

- Provide the keywords as a JSON array of strings.
- Ensure all words are in their base (lemmatized) form.
- If you cannot extract any relevant keywords, respond exactly with 'None'.
- Do not include any additional text or explanation.

**Keywords**:
"""
)

SIMILAR_KEYWORDS_PROMPT = Template(
    """
You are tasked with finding additional keywords that are semantically similar to the provided keywords, using only the **Context** below.

---

**Provided Keywords**: $rule

---

**Context**:
$chunks

---

**Instructions**:

- Provide the similar keywords as a JSON array of strings.
- Only include words that are present in the context and are semantically related to the provided keywords.
- If you cannot find any similar keywords in the context, respond exactly with 'None'.
- Do not include any additional text or explanation.

**Similar Keywords**:
"""
)

DECOMPOSE_QUERY_PROMPT = Template(
    """
You are tasked with decomposing the following question into simpler, relevant sub-questions that capture different aspects of the original question.

---

**Original Question**: $query

---

**Instructions**:

- Provide up to 3 sub-questions as a JSON array of strings.
- If the question is already simple or cannot be decomposed, respond exactly with 'None'.
- Do not include any additional text or explanation.

**Sub-Questions**:
"""
)

SCHEMA_PROMPT = Template(
    """
Given the information about columns in a knowledge table, generate a schema that includes relationships between the columns if relevant.

---

**Documents**: $documents

**Columns**: $columns

**Available Column Names**: $entity_types

---

**Instructions**:

- Use **only** the exact column names provided in `$entity_types`.
- For each relationship, create an object with `"head"`, `"relation"`, and `"tail"` fields.
- The `"head"` and `"tail"` must be one of the provided column names.
- Create meaningful `"relation"` names based on the column information and questions.
- Do not use any names not in the provided column list.
- If you cannot generate any meaningful relationships, respond exactly with `"None"`.
- Do not include any additional text or explanation.

**Schema Relationships**:
"""
)
