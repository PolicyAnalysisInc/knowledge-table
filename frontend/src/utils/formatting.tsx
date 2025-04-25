import React from 'react';
import { Text, HoverCard, ScrollArea } from '@mantine/core';
import { Chunk } from '../config/store';

// --- Stringify Answer --- (Copied from DetailsModal)
export function stringifyAnswer(ans: any): string {
    if (ans === null || ans === undefined) return String(ans);
    if (typeof ans === 'object' && ans !== null && 'answer' in ans) {
        const actualAnswer = ans.answer;
        if (actualAnswer === null || actualAnswer === undefined) return String(actualAnswer);
        if (typeof actualAnswer === 'object') return JSON.stringify(actualAnswer);
        return String(actualAnswer);
    }
    if (typeof ans === 'object') return JSON.stringify(ans);
    return String(ans);
}

// --- Format Reasoning --- (Copied and adapted from DetailsModal)
export function formatReasoning(reasoning: string | null | undefined, chunks: Chunk[]): React.ReactNode {
  if (!reasoning) return <Text c="dimmed" fs="italic">No reasoning provided.</Text>;

  const parts: React.ReactNode[] = [];
  const citedOrder: number[] = []; 
  let lastIndex = 0;

  const citationRegex = /\[\s*chunk_(\d+)\s*\]/g;

  let match;
  while ((match = citationRegex.exec(reasoning)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`text-${lastIndex}`}>{reasoning.substring(lastIndex, match.index)}</span>);
    }

    const chunkIndex = parseInt(match[1], 10);
    let footnoteIndex = citedOrder.indexOf(chunkIndex);

    if (footnoteIndex === -1) {
      citedOrder.push(chunkIndex);
      footnoteIndex = citedOrder.length - 1;
    }

    const footnoteNumber = footnoteIndex + 1;
    const chunk = chunks[chunkIndex];

    if (chunk) {
      parts.push(
        <HoverCard key={`cite-${match.index}`} width={320} shadow="md" withArrow position="top" withinPortal openDelay={200} closeDelay={100}>
          <HoverCard.Target>
            <Text component="sup" size="xs" c="blue.6" fw={500} style={{ cursor: 'pointer', textDecoration: 'underline'}}>
              {footnoteNumber}
            </Text>
          </HoverCard.Target>
          <HoverCard.Dropdown p="xs">
            <Text size="sm" fw={600} mb={4}>Chunk {chunkIndex} (Page {chunk.page})</Text>
            <ScrollArea.Autosize mah={160} type="auto">
              <Text size="sm">{chunk.content}</Text>
            </ScrollArea.Autosize>
          </HoverCard.Dropdown>
        </HoverCard>
      );
    } else {
      parts.push(
        <Text key={`cite-${match.index}`} component="sup" c="red.6" title={`Chunk ${chunkIndex} not found`}>
          ?
        </Text>
      );
    }

    lastIndex = citationRegex.lastIndex;
  }

  if (lastIndex < reasoning.length) {
    parts.push(<span key={`text-${lastIndex}`}>{reasoning.substring(lastIndex)}</span>);
  }

  return <>{parts}</>;
} 