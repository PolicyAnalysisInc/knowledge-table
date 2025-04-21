import { useMemo } from "react";
import { Blockquote, Modal, Stack, Text, Alert, Divider, Title } from "@mantine/core";
import { IconInfoCircle } from "@tabler/icons-react";
import { isEmpty, pick, values } from "lodash-es";
import { z } from "zod";
import { useStore, getCellKey } from "@config/store";
import { chunkSchema } from "@config/api";

type Chunk = z.infer<typeof chunkSchema>;

export function KtChunks() {
  const allChunks = useStore(store => store.getTable().chunks);
  const openedChunksKeys = useStore(store => store.getTable().openedChunks);
  const allCitedIndices = useStore(store => store.getTable().citedIndices);

  // Get chunks and indices for the currently opened cells
  // Assuming openedChunksKeys is usually just one key, but handle multiple if needed.
  // For simplicity, we'll just use the first key if multiple are selected.
  const currentCellKey = openedChunksKeys.length > 0 ? openedChunksKeys[0] : null;
  const currentChunks: Chunk[] = currentCellKey ? (allChunks[currentCellKey] ?? []) : [];
  const currentCitedIndices: number[] = currentCellKey ? (allCitedIndices[currentCellKey] ?? []) : [];

  // Partition chunks based on cited indices
  const citedChunks: Chunk[] = [];
  const otherChunks: Chunk[] = [];

  if (currentChunks.length > 0) {
    const citedIndexSet = new Set(currentCitedIndices);
    currentChunks.forEach((chunk, index) => {
      if (citedIndexSet.has(index)) {
        citedChunks.push(chunk);
      } else {
        otherChunks.push(chunk);
      }
    });
  }

  return (
    <Modal
      size="xl"
      title="Retrieved Chunks"
      opened={!isEmpty(openedChunksKeys)}
      onClose={() => useStore.getState().closeChunks()}
    >
      {currentChunks.length === 0 ? (
        <Text>No chunks found for the selected cell(s).</Text>
      ) : (
        <Stack>
          {citedChunks.length > 0 && (
            <>
              <Alert
                variant="light"
                color="blue"
                title="Cited by Model"
                icon={<IconInfoCircle />}
                mb="md"
              >
                The following text chunk(s) were cited by the model as the basis for the generated answer.
              </Alert>
              {citedChunks.map((chunk: Chunk, index) => (
                <Blockquote
                  key={`cited-${index}`}
                  style={{ backgroundColor: "var(--mantine-color-blue-light)" }}
                  p="sm"
                >
                  {chunk.content}
                  <Text size="xs" c="dimmed" mt="xs">Page: {chunk.page}</Text>
                </Blockquote>
              ))}
            </>
          )}

          {otherChunks.length > 0 && (
            <>
              {(citedChunks.length > 0) && <Divider my="lg" label="Other Retrieved Chunks (Not Cited)" labelPosition="center" />}
              {(citedChunks.length === 0) && 
                <Title order={5} mb="md">Retrieved Chunks (Not Cited)</Title>
              }
              {otherChunks.map((chunk: Chunk, index) => (
                <Blockquote key={`other-${index}`} p="sm">
                  {chunk.content}
                  <Text size="xs" c="dimmed" mt="xs">Page: {chunk.page}</Text>
                </Blockquote>
              ))}
            </>
          )}
        </Stack>
      )}
    </Modal>
  );
}
