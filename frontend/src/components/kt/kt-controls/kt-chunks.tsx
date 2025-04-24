import { useMemo, useState } from "react";
import {
  Blockquote,
  Modal,
  Stack,
  Text,
  Button,
  Group
} from "@mantine/core";
import { isEmpty, pick, values } from "lodash-es";
import { useStore, Chunk } from "@config/store";

export function KtChunks() {
  const [showAll, setShowAll] = useState(false);
  const allChunksMap = useStore(store => store.getTable().chunks);
  const allCitationsMap = useStore(store => store.getTable().citations);
  const openedChunkKeys = useStore(store => store.getTable().openedChunks);

  const {
    displayChunks,
    totalRetrievedCount,
    totalCitedCount,
    hasAnyCitations,
    canShowMore // Derived based on counts and showAll state
  } = useMemo(() => {
    let retrievedChunks: Chunk[] = [];
    let citedChunksFromAllCells: Chunk[] = [];
    let _totalRetrievedCount = 0;
    let _hasAnyCitations = false;

    openedChunkKeys.forEach(key => {
      const chunksForKey = allChunksMap[key] || [];
      const citationsForKey = allCitationsMap[key] || []; // This is now number[] | undefined
      _totalRetrievedCount += chunksForKey.length;
      retrievedChunks.push(...chunksForKey); // Add all chunks for potential 'showAll'

      if (!isEmpty(citationsForKey)) {
        // Directly use the numbers from citationsForKey
        const citationIndices = new Set<number>(citationsForKey);

        if (citationIndices.size > 0) {
          _hasAnyCitations = true;
          // Filter chunks *for this specific key* based on *its* citations
          const citedForKey = chunksForKey.filter((_, index) =>
            citationIndices.has(index)
          );
          citedChunksFromAllCells.push(...citedForKey);
        } else {
           // Citations array might be technically present but empty
        }
      } else {
         // No citations array for this cell key
      }
    });

    // Deduplicate chunks (important if same chunk appears for multiple selected cells)
    // Using simple stringify for comparison, might need a more robust ID later if available
    const uniqueRetrievedChunks = Array.from(new Map(retrievedChunks.map(c => [JSON.stringify(c), c])).values());
    const uniqueCitedChunks = Array.from(new Map(citedChunksFromAllCells.map(c => [JSON.stringify(c), c])).values());

    const _totalCitedCount = uniqueCitedChunks.length;
    const _displayChunks = _hasAnyCitations && !showAll ? uniqueCitedChunks : uniqueRetrievedChunks;
    const _canShowMore = _hasAnyCitations && _totalCitedCount < _totalRetrievedCount;


    return {
      displayChunks: _displayChunks,
      totalRetrievedCount: _totalRetrievedCount, // Use original count before dedupe for button label
      totalCitedCount: _totalCitedCount,         // Use unique count for button label
      hasAnyCitations: _hasAnyCitations,
      canShowMore: _canShowMore
    };
  }, [allChunksMap, allCitationsMap, openedChunkKeys, showAll]); // Add showAll dependency


  return (
    <Modal
      size="xl"
      title="Chunks"
      opened={!isEmpty(openedChunkKeys)}
      onClose={() => {
        useStore.getState().closeChunks();
        setShowAll(false);
      }}
    >
      {totalRetrievedCount === 0 ? ( // Check total count before dedupe
        <Text>No chunks found for selected cells</Text>
      ) : (
        <Stack>
          {displayChunks.map((chunk, index) => (
            // Use a more stable key if possible, e.g., chunk ID if it exists
            <Blockquote key={`${chunk.page}-${index}`} cite={`Page ${chunk.page}`}>
              {chunk.content}
            </Blockquote>
          ))}
          {canShowMore && (
            <Group justify="center" mt="sm">
              <Button variant="subtle" onClick={() => setShowAll(!showAll)}>
                {showAll
                  ? `Show only ${totalCitedCount} cited chunks` // Use unique cited count
                  : `Show all ${totalRetrievedCount} retrieved chunks`} // Use total retrieved count
              </Button>
            </Group>
          )}
          {!hasAnyCitations && totalRetrievedCount > 0 && ( // Check total count before dedupe
             <Text size="sm" c="dimmed" ta="center" mt="sm">
               Showing all {totalRetrievedCount} retrieved chunks (no specific citations available).
             </Text>
          )}
        </Stack>
      )}
    </Modal>
  );
}
