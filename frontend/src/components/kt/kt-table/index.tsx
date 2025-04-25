import { useMemo } from "react";
import { Column, ReactGrid, Row } from "@silevis/reactgrid";
import { BoxProps, ScrollArea } from "@mantine/core";
import {
  Cell,
  handleCellChange,
  handleContextMenu,
  HEADER_ROW_ID,
  SOURCE_COLUMN_ID
} from "./index.utils";
import {
  KtCell,
  KtCellTemplate,
  KtColumnCell,
  KtColumnCellTemplate,
  KtRowCellTemplate
} from "./kt-cells";
import { useStore } from "@config/store";
import { cn } from "@utils/functions";
import classes from "./index.module.css";

export function KtTable(props: BoxProps) {
  const columns = useStore(state => state.getTable().columns);
  const rows = useStore(state => state.getTable().rows);

  // --- DEBUG LOGGING START ---
  console.log("[KtTable] Received rows from store:", rows);
  // If you know the specific row/column ID being changed, log it:
  // const testRowId = '...'; // Replace with actual ID
  // const testColId = '...'; // Replace with actual ID
  // const testRow = rows.find(r => r.id === testRowId);
  // if (testRow) {
  //   console.log(`[KtTable] Value for ${testRowId} / ${testColId}:`, testRow.cells[testColId]);
  // }
  // --- DEBUG LOGGING END ---

  const visibleColumns = useMemo(
    () => columns.filter(column => !column.hidden),
    [columns]
  );
  const visibleRows = useMemo(() => rows.filter(row => !row.hidden), [rows]);

  const gridColumns = useMemo<Column[]>(
    () => [
      { columnId: SOURCE_COLUMN_ID, width: 260 },
      ...visibleColumns.map(column => ({
        columnId: column.id,
        width: column.width,
        resizable: true
      }))
    ],
    [visibleColumns]
  );

  const gridRows = useMemo<Row<Cell>[]>(
    () => {
      console.log("[KtTable] Recalculating gridRows..."); // Log when memo recalculates
      return [
        {
          rowId: HEADER_ROW_ID,
          cells: [
            { type: "header", text: "" },
            ...visibleColumns.map<KtColumnCell>(column => ({
              type: "kt-column",
              column
            }))
          ]
        },
        ...visibleRows.map<Row<Cell>>(row => ({
          rowId: row.id,
          key: `${row.id}-${JSON.stringify(row.cells)}`,
          height: 48,
          cells: [
            { type: "kt-row", row },
            ...visibleColumns.map<KtCell>(column => {
              // --- DEBUG LOGGING START ---
              // Log the specific cell value being passed to the grid cell object
              // if (row.id === testRowId && column.id === testColId) { 
              //   console.log(`[KtTable gridRows Memo] Mapping cell value for ${row.id}/${column.id}:`, row.cells[column.id]);
              // }
              // --- DEBUG LOGGING END ---
              return {
                type: "kt-cell",
                column,
                row,
                cell: row.cells[column.id]
              };
            })
          ]
        }))
      ];
    },
    [visibleRows, visibleColumns]
  );

  return (
    <ScrollArea
      {...props}
      pb="md"
      className={cn(classes.reactGridWrapper, props.className)}
    >
      <ReactGrid
        enableRangeSelection
        enableColumnSelection
        enableRowSelection
        minColumnWidth={100}
        columns={gridColumns}
        rows={gridRows}
        onContextMenu={handleContextMenu}
        onCellsChanged={handleCellChange}
        onColumnResized={(columnId, width) =>
          useStore.getState().editColumn(String(columnId), { width })
        }
        customCellTemplates={{
          "kt-cell": new KtCellTemplate(),
          "kt-column": new KtColumnCellTemplate(),
          "kt-row": new KtRowCellTemplate()
        }}
      />
    </ScrollArea>
  );
}
