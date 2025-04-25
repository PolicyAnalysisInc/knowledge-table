import React from 'react';
import { Modal, Grid, Card, Text, ScrollArea, Alert, Badge, Box, useMantineColorScheme } from '@mantine/core';
import { useStore, CellDetails, Chunk, CellKey } from '@config/store';
// Update import path
import { formatReasoning, stringifyAnswer } from '@utils/formatting';

export function CompareAnswersModal() {
  // Get color scheme
  const { colorScheme } = useMantineColorScheme();

  // Select necessary state and actions
  const {
    compareModalOpen,
    compareModalCellKey,
    closeCompareModal,
    selectAnswerFromComparison,
    // Select cellDetails directly based on the key
    cellDetails 
  } = useStore(state => {
    const key = state.compareModalCellKey;
    const table = key ? state.tables.find(t => t.id === state.activeTableId) : undefined;
    // Find the specific cell details if key and table exist
    const details = key && table ? table.cellDetails[key as keyof typeof table.cellDetails] : undefined; 
    
    return {
      compareModalOpen: state.compareModalOpen,
      compareModalCellKey: key,
      closeCompareModal: state.closeCompareModal,
      selectAnswerFromComparison: state.selectAnswerFromComparison,
      cellDetails: details // Directly return the details object
    };
  });

  // No need for useMemo anymore, cellDetails comes directly from the store selector
  // const cellDetails: CellDetails | undefined = React.useMemo(() => { ... }, [compareModalCellKey, getTable]);

  // Prepare data for rendering (use cellDetails directly from store)
  const allResponses = cellDetails?.all_responses || [];
  const chunks = cellDetails?.chunks || [];

  // Filter valid responses (must have reasoning for this view)
  const validResponses = (allResponses || []) 
    .filter(
      resp => resp !== null && typeof resp === 'object' && resp.reasoning !== undefined 
    ) as NonNullable<CellDetails['all_responses']>[number][];

  // Handler for clicking a non-selected card
  const handleCardClick = (responseIndex: number) => {
    if (compareModalCellKey) {
      selectAnswerFromComparison(compareModalCellKey, responseIndex);
      // Optional: close modal after selection? 
      // closeCompareModal(); 
    }
  };

  // Add logging to check the filter result (useful for debugging)
  console.log('Compare Modal - Cell Details from Store:', cellDetails);
  console.log('Compare Modal - Filtered Valid Responses:', validResponses);

  return (
    <Modal
      opened={compareModalOpen}
      onClose={closeCompareModal}
      title="Compare LLM Answers"
      size="90%" // Use a larger percentage size
      scrollAreaComponent={ScrollArea.Autosize}
      centered
    >
      {!cellDetails ? (
        <Alert color="gray" title="No Data">
          Could not load comparison data for the selected cell.
        </Alert>
      ) : (
        // Render the grid only if there are valid responses
        validResponses.length > 0 && (
          <Grid gutter="md">
            {validResponses.map((response, index) => {
              // Add null check for safety
              if (!response) return null;

              const isCurrent = response.is_selected_answer === true;
              const numCols = Math.min(validResponses.length, 4);
              const span = Math.max(12 / numCols, 3);

              return (
                <Grid.Col span={span} key={index}>
                  <Card
                    shadow={isCurrent ? "md" : "sm"}
                    padding="lg"
                    radius="md"
                    withBorder
                    onClick={() => !isCurrent && handleCardClick(index)}
                    style={theme => ({
                      borderColor: isCurrent ? theme.colors.blue[6] : theme.colors.gray[3],
                      borderWidth: isCurrent ? '2px' : '1px',
                      height: '100%',
                      cursor: isCurrent ? 'default' : 'pointer',
                      transition: 'border-color 0.2s ease',
                      '&:hover': !isCurrent ? {
                        borderColor: theme.colors.blue[4]
                      } : {}
                    })}
                  >
                    {isCurrent && (
                      <Badge color="blue" variant="filled" style={{ position: 'absolute', top: '10px', right: '10px' }}>
                        Selected
                      </Badge>
                    )}
                    {/* Model Name as Header */}
                    <Text fw={500} size="lg" mb="md" mr={80}>
                      {response.model_name || 'Unknown Model'} {/* Fallback text */}
                    </Text>

                    {/* Answer */}
                    <Text size="sm" mb="md">
                      <Text span fw={500}>Answer:</Text> {stringifyAnswer(response.answer)}
                    </Text>

                    {/* Reasoning Section */}
                    <Box 
                      p="xs" 
                      mb="sm" 
                      style={theme => ({ 
                        backgroundColor: colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[0],
                        borderRadius: theme.radius.sm 
                      })}
                    >
                      <Text fw={500} size="sm" mb="xs">Reasoning:</Text>
                      <ScrollArea.Autosize mah={250} type="auto" offsetScrollbars scrollbarSize={8}>
                        <Text size="sm" c="dimmed">
                          {formatReasoning(response.reasoning, chunks)}
                        </Text>
                      </ScrollArea.Autosize>
                    </Box>
                  </Card>
                </Grid.Col>
              );
            })}
          </Grid>
        )
      )}
    </Modal>
  );
} 