import React from 'react';
import { Modal, Grid, Card, Text, ScrollArea, Alert, Badge, Box, useMantineColorScheme } from '@mantine/core';
import { useStore, CellDetails, Chunk } from '@config/store';
// Update import path
import { formatReasoning, stringifyAnswer } from '@utils/formatting';

export function CompareAnswersModal() {
  // Get color scheme
  const { colorScheme } = useMantineColorScheme();

  const {
    compareModalOpen,
    compareModalCellKey,
    closeCompareModal,
    getTable
  } = useStore(state => ({
    compareModalOpen: state.compareModalOpen,
    compareModalCellKey: state.compareModalCellKey,
    closeCompareModal: state.closeCompareModal,
    getTable: state.getTable // Need this to access cellDetails
  }));

  // Get cell details based on the key from the active table
  const cellDetails: CellDetails | undefined = React.useMemo(() => {
    if (!compareModalCellKey) return undefined;
    try {
      const table = getTable(); // Get current table state
      return table.cellDetails[compareModalCellKey];
    } catch (error) {
      console.error("Error getting table for compare modal:", error);
      return undefined;
    }
  }, [compareModalCellKey, getTable]);

  // Prepare data for rendering
  const currentAnswer = cellDetails?.answer?.answer;
  const allResponses = cellDetails?.all_responses || [];
  const chunks = cellDetails?.chunks || [];

  // Filter valid responses (must have reasoning for this view)
  const validResponses = (allResponses || []) // Ensure allResponses is an array
    .filter(
      // Simplified filter: check for non-null object with reasoning
      resp => resp !== null && typeof resp === 'object' && resp.reasoning !== undefined 
    ) as NonNullable<CellDetails['all_responses']>[number][];

  // Add logging to check the filter result
  console.log('Filtered validResponses for Compare Modal:', validResponses);

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

              // Compare answer, reasoning, and citations to uniquely identify the selected response
              const isCurrent = 
                // Check the backend flag
                response.is_selected_answer === true;
              
              // Calculate column span - show max 4 side-by-side
              const numCols = Math.min(validResponses.length, 4);
              const span = Math.max(12 / numCols, 3); // Ensure minimum span of 3

              return (
                <Grid.Col span={span} key={index}>
                  <Card
                    shadow={isCurrent ? "md" : "sm"}
                    padding="lg"
                    radius="md"
                    withBorder
                    style={{
                      borderColor: isCurrent ? 'var(--mantine-color-blue-filled)' : undefined,
                      borderWidth: isCurrent ? '2px' : '1px',
                      height: '100%' // Ensure cards in a row have same height
                    }}
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
                        // Remove border
                        // border: `1px solid ${colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[3]}`, 
                        // Add background color for "well" effect
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