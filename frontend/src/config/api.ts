import { z } from "zod";
import {
  isArray,
  isNil,
  isPlainObject,
  isString,
  mapValues,
  omit
} from "lodash-es";
import {
  AnswerTableColumn,
  AnswerTableGlobalRule,
  AnswerTableRow
} from "./store";

// Upload file

export const documentSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    author: z.string(),
    tag: z.string(),
    page_count: z.number()
  })
  .strict();

export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const result = await fetch("http://localhost:8000/api/v1/document", {
    method: "POST",
    body: formData
  });
  return documentSchema.parse(await result.json());
}

// Delete document

export async function deleteDocument(id: string) {
  await fetch(`http://localhost:8000/api/v1/document/${id}`, {
    method: "DELETE"
  });
}

// Run query

export const chunkSchema = z
  .object({
    content: z.string(),
    page: z.number()
  })
  .strict();

export const answerSchema = z.union([
  z.null(),
  z.number(),
  z.string(),
  z.boolean(),
  z.array(z.number()),
  z.array(z.string())
]);

// Define the schema for the *base* fields shared by all responses
const baseResponseFieldsSchema = z.object({
  confidence: z.number().int().min(1).max(10).nullable(),
  reasoning: z.string().nullable(), 
  citations: z.array(z.number().int()),
  is_selected_answer: z.boolean().optional().nullable(),
  model_name: z.string().optional().nullable()
});

// Define schemas for each concrete answer type, extending the base fields
const boolResponseSchema = baseResponseFieldsSchema.extend({ answer: z.boolean().nullable() });
const intResponseSchema = baseResponseFieldsSchema.extend({ answer: z.number().int().nullable() });
const numberResponseSchema = baseResponseFieldsSchema.extend({ answer: z.number().nullable() });
const strResponseSchema = baseResponseFieldsSchema.extend({ answer: z.string().nullable() });
const intArrayResponseSchema = baseResponseFieldsSchema.extend({ answer: z.array(z.number().int()).nullable() });
const numberArrayResponseSchema = baseResponseFieldsSchema.extend({ answer: z.array(z.number()).nullable() });
const strArrayResponseSchema = baseResponseFieldsSchema.extend({ answer: z.array(z.string()).nullable() });

// Create a Zod Union schema for any possible response type within all_responses
const anyResponseSchema = z.union([
  boolResponseSchema,
  intResponseSchema,
  numberResponseSchema,
  strResponseSchema,
  intArrayResponseSchema,
  numberArrayResponseSchema,
  strArrayResponseSchema,
  // Add others if necessary (e.g., keywords, schema, subqueries)
]);

export const resolvedEntitySchema = z.object({
  original: z.union([z.string(), z.array(z.string())]),
  resolved: z.union([z.string(), z.array(z.string())]),
  source: z.object({
    type: z.string(),
    id: z.string()
  }),
  entityType: z.string(),
  citations: z.array(z.number()),
  reasoning: z.string().nullable().optional(),
});

// Update the resolved entities schema to match backend format
export const resolvedEntitiesSchema = z.union([
  z.array(resolvedEntitySchema),
  z.null(),
  z.undefined()
]);

// Update the query response schema
const queryResponseSchema = z.object({
  answer: z.object({ answer: answerSchema }),
  chunks: z.array(chunkSchema),
  resolved_entities: resolvedEntitiesSchema,
  citations: z.array(z.number()),
  reasoning: z.string().nullable().optional(),
  all_responses: z.array(z.union([anyResponseSchema, z.null()])).nullable().optional(),
});

// Update the runQuery function to transform the data format
export async function runQuery(
  row: AnswerTableRow,
  column: AnswerTableColumn,
  globalRules: AnswerTableGlobalRule[]
) {
  if (!column.entityType.trim() || !column.generate) {
    throw new Error(
      "Row or column doesn't allow running query (missing row source data or column is empty or has generate set to false)"
    );
  }
  const rules = [
    ...column.rules,
    ...globalRules
      .filter(rule => rule.entityType.trim() === column.entityType.trim())
      .map(r => omit(r, "id", "entityType"))
  ];
  
  const result = await fetch("http://localhost:8000/api/v1/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      document_id: row.sourceData?.document?.id
        ? row.sourceData.document.id
        : "00000000000000000000000000000000",
      prompt: {
        id: column.id,
        entity_type: column.entityType,
        query: column.query,
        type: column.type,
        rules
      }
    })
  });
  
  const response = await result.json();
  console.log('Raw API Response:', response);
  
  const parsed = queryResponseSchema.parse(response);
  console.log('Parsed Response:', parsed);
  
  // Update resolved entities transformation to handle the new format
  const resolvedEntities = parsed.resolved_entities?.map(entity => ({
    original: entity.original,
    resolved: entity.resolved,
    source: entity.source,
    entityType: entity.entityType,
    fullAnswer: parsed.answer.answer as string
  })) ?? null;
  
  console.log('Transformed Resolved Entities:', resolvedEntities);

  return {
    answer: parsed.answer,
    chunks: parsed.chunks,
    resolvedEntities,
    citations: parsed.citations,
    reasoning: parsed.reasoning,
    all_responses: parsed.all_responses,
  };
}

// Export triples

export async function exportTriples(tableData: any) {
  function stringifyDeep(value: any): any {
    if (isNil(value)) {
      return "";
    } else if (isString(value)) {
      return value;
    } else if (isArray(value)) {
      return value.map(stringifyDeep);
    } else if (isPlainObject(value)) {
      return mapValues(value, stringifyDeep);
    } else {
      return String(value);
    }
  }

  return fetch("http://localhost:8000/api/v1/graph/export-triples", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(stringifyDeep(tableData))
  }).then(r => r.blob());
}
