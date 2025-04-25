import React from 'react';
import { Modal, Tabs, Text, ScrollArea, Paper, Alert, Button, HoverCard } from '@mantine/core';
import { useStore, CellDetails, Chunk } from '@config/store';
// Update import path
import { formatReasoning, stringifyAnswer } from '@utils/formatting';

// --- Remove local helper function definitions ---
/*
function formatReasoning(...) { ... }
function stringifyAnswer(...) { ... }
*/

export function DetailsModal() {
  const {
    detailsModalOpen,
    detailsModalCellData,
    closeDetailsModal
  } = useStore(state => ({
    detailsModalOpen: state.detailsModalOpen,
    detailsModalCellData: state.detailsModalCellData,
    closeDetailsModal: state.closeDetailsModal,
  }));

  // Call hooks unconditionally at the top level
  const [showAllChunks, setShowAllChunks] = React.useState(false);

  // Early return if no data
  if (!detailsModalCellData) {
    return null;
  }

  // Safely access data from CellValue (which is the full query response object)
  const reasoning = detailsModalCellData.reasoning;
  // The actual answer is nested within the 'answer' object
  const currentAnswer = detailsModalCellData.answer?.answer;
  const chunks = detailsModalCellData.chunks || [];
  const citations = detailsModalCellData.citations || [];
  // Access all_responses if it exists
  const allResponses = detailsModalCellData.all_responses || []; // Default to empty array

  // Filter responses to only include those with actual content (not null/error objects)
  // And extract just the 'answer' part for display
  const validAlternativeAnswers = allResponses
    .filter(resp => resp !== null && typeof resp === 'object' && 'answer' in resp && resp.answer !== undefined)
    .map(resp => (resp as any).answer);

  // Identify cited chunks
   const citedChunks = chunks.filter((_, index) => citations.includes(index));
   const displayChunks = showAllChunks ? chunks : citedChunks;


  return (
    <Modal
      opened={detailsModalOpen}
      onClose={closeDetailsModal}
      title="Cell Details"
      size="xl" // Larger modal size
      scrollAreaComponent={ScrollArea.Autosize} // Make modal content scrollable if needed
      centered // Center the modal
    >
      {/* Optional: Add a small description */}
      <Text size="sm" c="dimmed" mb="lg">
        Detailed information about the selected cell's content and generation process.
      </Text>

      <Tabs defaultValue="reasoning">
        <Tabs.List grow>
          <Tabs.Tab value="reasoning">Reasoning</Tabs.Tab>
          <Tabs.Tab value="answers">
             Answers ({validAlternativeAnswers.length > 0 ? validAlternativeAnswers.length : (currentAnswer !== undefined ? 1 : 0)})
          </Tabs.Tab>
          <Tabs.Tab value="chunks">Chunks ({chunks.length})</Tabs.Tab>
        </Tabs.List>

        {/* Reasoning Panel */}
        <Tabs.Panel value="reasoning" pt="xs">
          <ScrollArea.Autosize mah={400} type="auto">
             <Paper p="sm" withBorder radius="md" shadow="xs">
                 {formatReasoning(reasoning, chunks)}
             </Paper>
          </ScrollArea.Autosize>
        </Tabs.Panel>

        {/* Answers Panel */}
        <Tabs.Panel value="answers" pt="xs">
           <ScrollArea.Autosize mah={400} type="auto">
             {(() => {
               // Filter valid responses from all_responses
               const validAlternativeAnswers = allResponses
                 .filter(resp => resp !== null && typeof resp === 'object' && 'answer' in resp && resp.answer !== undefined)
                 .map(resp => (resp as any).answer);

               // Check if current answer exists
               const hasCurrentAnswer = currentAnswer !== undefined;

               // Render current answer if it exists
               const currentAnswerElement = hasCurrentAnswer ? (
                 <Paper
                    key="current-answer"
                    p="sm"
                    mb="xs"
                    withBorder
                    radius="md"
                    shadow="xs"
                    bg={'var(--mantine-color-blue-light)'}
                 >
                    <Text size="sm" fw={500}>
                        {stringifyAnswer(currentAnswer)}
                        <Text span c="blue" size="xs" ml={5}>(Current Selection)</Text>
                    </Text>
                 </Paper>
               ) : null;

               // Render alternative answers (all valid ones from the list)
               const alternativeAnswerElements = validAlternativeAnswers.map((altAns, index) => {
                 // Don't re-render if it's identical to the current answer we already displayed
                 // (This avoids showing the exact same item twice if the judged answer is also in the list)
                 if (hasCurrentAnswer && stringifyAnswer(altAns) === stringifyAnswer(currentAnswer)) {
                   return null;
                 }
                 return (
                   <Paper
                      key={`alt-${index}`}
                      p="sm"
                      mb="xs"
                      withBorder
                      radius="md"
                      shadow="none"
                   >
                      <Text size="sm" fw={400}>
                          {stringifyAnswer(altAns)}
                          {/* Optional: Indicate if it matches current */}
                          {/* {stringifyAnswer(altAns) === stringifyAnswer(currentAnswer) && (\n                              <Text span c=\"dimmed\" size=\"xs\" ml={5}>(Matches Selection)</Text>\n                          )} */}
                      </Text>
                   </Paper>
                 );
               });

               // Combine and render, or show alert if nothing is displayable
               const allElements = [currentAnswerElement, ...alternativeAnswerElements].filter(Boolean);

               if (allElements.length > 0) {
                 return allElements;
               } else {
                 return (
                    <Alert color="gray" title="No Answers Recorded">
                       No valid answers were generated or recorded for this cell.
                    </Alert>
                 );
               }
             })()}
           </ScrollArea.Autosize>
        </Tabs.Panel>

        {/* Chunks Panel */}
        <Tabs.Panel value="chunks" pt="xs">
           <ScrollArea.Autosize mah={400} type="auto">
             <Text fw={500} mb="xs">
                 {showAllChunks ? 'All Retrieved Chunks' : 'Cited Chunks'} ({displayChunks.length})
             </Text>
             {displayChunks.length > 0 ? (
                 displayChunks.map((chunk, index) => {
                     // Find the original index in the main chunks array for display
                     const originalIndex = chunks.findIndex(c => c === chunk);
                     return (
                         <Paper key={originalIndex} p="xs" mb="xs" withBorder bg="var(--mantine-color-gray-light)">
                             <Text size="sm" fw={600} mb={4}>Chunk {originalIndex} (Page {chunk.page})</Text>
                             <Text size="sm">{chunk.content}</Text>
                         </Paper>
                     );
                 })
             ) : (
                 <Text size="sm" c="dimmed" fs="italic">
                     {showAllChunks ? 'No chunks were retrieved for this query.' : 'No specific chunks were cited in the reasoning.'}
                 </Text>
             )}
             {/* Toggle Button Logic */}
             {chunks.length > citedChunks.length && !showAllChunks && (
                 <Button variant="subtle" size="xs" onClick={() => setShowAllChunks(true)} mt="xs" p={0} h="auto">
                     Show all {chunks.length} retrieved chunks...
                 </Button>
             )}
             {showAllChunks && chunks.length > 0 && citedChunks.length !== chunks.length && (
                 <Button variant="subtle" size="xs" onClick={() => setShowAllChunks(false)} mt="xs" p={0} h="auto">
                     Show only cited chunks ({citedChunks.length})
                 </Button>
             )}
           </ScrollArea.Autosize>
        </Tabs.Panel>
      </Tabs>
    </Modal>
  );
} 