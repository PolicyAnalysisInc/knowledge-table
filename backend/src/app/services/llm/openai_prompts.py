"""The prompts for the language model."""

from string import Template

# Shared confidence instruction snippet
CONFIDENCE_INSTRUCTION = "- Include a `confidence` score (integer from 1 to 10, 1 being lowest confidence, 10 being highest) indicating how certain you are about the answer based *only* on the provided context."
REASONING_INSTRUCTION = "- Include a `reasoning` field (string) explaining the step-by-step process and the parts of the context you used to arrive at your answer."

BASE_PROMPT = Template(
    """
You are an expert assistant whose job is to answer the following question using **only** the information provided in the **Context**. Do not use any prior knowledge or external information.

Your response MUST be a JSON object with three fields: 'answer', 'confidence', and 'reasoning'.

---

**Question**: $query

---

**Context**:
$chunks

---

$format_specific_instructions

**Instructions**:

- Provide your answer based strictly on the given context.
- Be concise and accurate.
- Do not include any introductory or concluding remarks.
- If the answer is not present in the context, the 'answer' field should be exactly `null`.
- Do not include JSON or any code inside strins in the response. Any string responses should be the most direct human readable answer.
{confidence_instruction}
{reasoning_instruction}

**Answer** (JSON object with 'answer', 'confidence', and 'reasoning' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION, reasoning_instruction=REASONING_INSTRUCTION)
)

INFERRED_BASE_PROMPT = Template(
    """
Answer the following question following the formatting instructions at the bottom. Do not include quotes, formatting, or any explanation or extra information. Just answer the question.

Your response MUST be a JSON object with three fields: 'answer', 'confidence', and 'reasoning'.

**Question**: $query

$format_specific_instructions

{confidence_instruction}
{reasoning_instruction}

**Answer** (JSON object with 'answer', 'confidence', and 'reasoning' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION, reasoning_instruction=REASONING_INSTRUCTION)
)

BOOL_INSTRUCTIONS = """
**Special Instructions for Boolean Questions**:

- If the question is asking for a verification or requires a boolean answer, the 'answer' field should contain `true` or `false`.
- If you cannot answer the question, the 'answer' field should be exactly `null`.
- Do not provide any explanations or additional information in the 'answer' field.
"""

STR_ARRAY_INSTRUCTIONS = Template(
    """
$str_rule_line
$int_rule_line

**Special Instructions for String Responses**:

- If the answer is a single string, the 'answer' field should contain a single string.
- If multiple strings are required, the 'answer' field should contain a JSON array of strings.
- If you cannot find an answer, the 'answer' field should be exactly `null`.
- Do not include any additional text or explanation in the 'answer' field.
"""
)

NUMBER_ARRAY_INSTRUCTIONS = Template(
    """
$int_rule_line

**Special Instructions for Numeric Responses**:

- If the answer is a single number, the 'answer' field should contain the number.
- If multiple numbers are required, the 'answer' field should contain a JSON array of numbers.
- If you cannot find an answer, the 'answer' field should be exactly `null`.
- Do not include any additional text or explanation in the 'answer' field.
"""
)

KEYWORD_PROMPT = Template(
    """
You are tasked with extracting the most relevant keywords from the following query. Focus on the main nouns and verbs that capture the essence of the query.

Your response MUST be a JSON object with two fields: 'keywords' and 'confidence'.

---

**Query**: $query

---

**Instructions**:

- Provide the keywords as a JSON array of strings in the 'keywords' field.
- Ensure all words are in their base (lemmatized) form.
- If you cannot extract any relevant keywords, the 'keywords' field should be exactly `null`.
- Do not include any additional text or explanation in the 'keywords' field.
{confidence_instruction}

**Keywords** (JSON object with 'keywords' and 'confidence' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION)
)

SIMILAR_KEYWORDS_PROMPT = Template(
    """
You are tasked with finding additional keywords that are semantically similar to the provided keywords, using only the **Context** below.

Your response MUST be a JSON object with two fields: 'keywords' and 'confidence'.

---

**Provided Keywords**: $rule

---

**Context**:
$chunks

---

**Instructions**:

- Provide the similar keywords as a JSON array of strings in the 'keywords' field.
- Only include words that are present in the context and are semantically related to the provided keywords.
- If you cannot find any similar keywords in the context, the 'keywords' field should be exactly `null`.
- Do not include any additional text or explanation in the 'keywords' field.
{confidence_instruction}

**Similar Keywords** (JSON object with 'keywords' and 'confidence' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION)
)

DECOMPOSE_QUERY_PROMPT = Template(
    """
You are tasked with decomposing the following question into simpler, relevant sub-questions that capture different aspects of the original question.

Your response MUST be a JSON object with two fields: 'sub_queries' and 'confidence'.

---

**Original Question**: $query

---

**Instructions**:

- Provide up to 3 sub-questions as a JSON array of strings in the 'sub_queries' field.
- If the question is already simple or cannot be decomposed, the 'sub_queries' field should be exactly `null`.
- Do not include any additional text or explanation in the 'sub_queries' field.
{confidence_instruction}

**Sub-Questions** (JSON object with 'sub_queries' and 'confidence' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION)
)

SCHEMA_PROMPT = Template(
    """
Given the information about columns in a knowledge table, generate a schema that includes relationships between the columns if relevant.

Your response MUST be a JSON object with two fields: 'relationships' and 'confidence'.

---

**Documents**: $documents

**Columns**: $columns

**Available Column Names**: $entity_types

---

**Instructions**:

- Use **only** the exact column names provided in `$entity_types`.
- For each relationship, create an object with `"head"`, `"relation"`, and `"tail"` fields. Place these relationship objects in a JSON array within the 'relationships' field.
- The `"head"` and `"tail"` must be one of the provided column names.
- Create meaningful `"relation"` names based on the column information and questions.
- Do not use any names not in the provided column list.
- If you cannot generate any meaningful relationships, the 'relationships' field should be exactly `null`.
- Do not include any additional text or explanation in the 'relationships' field.
{confidence_instruction}

**Schema Relationships** (JSON object with 'relationships' and 'confidence' fields):
""".format(confidence_instruction=CONFIDENCE_INSTRUCTION)
)

JUDGE_RESPONSE_PROMPT = Template(
    """
You are an expert evaluator. Your task is to analyze multiple responses generated for the same query and select the single best response based on accuracy, completeness, adherence to instructions (if any implied by the query), reasoning quality, and confidence score.

Original Query (including context/chunks):
---
$original_query
---

Candidate Responses (evaluate these):
---
$candidate_responses_json
---

Instructions:
1. Review each candidate response carefully.
2. Compare them against the original query, paying close attention to the context/chunks provided within the query.
3. Evaluate them based on accuracy, reasoning, and confidence **relative to the provided context**.
4. Choose the index (the 'index' field in the JSON above) of the single best response.
5. Respond with a JSON object containing ONLY the field 'answer', which MUST be the 0-based index of the best response you selected.

Respond ONLY with the JSON object: {{\"answer\": <chosen_index>}}
"""
)
