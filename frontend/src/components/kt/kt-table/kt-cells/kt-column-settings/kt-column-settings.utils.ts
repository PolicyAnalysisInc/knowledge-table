import { ReactNode } from "react";
import {
  IconAccessPoint,
  IconAccessPointOff,
  IconAlignJustified,
  IconCheckbox,
  IconHash,
  TablerIcon,
  IconCheck,
  IconList
} from "@tabler/icons-react";
import { AnswerTableColumn } from "@config/store";

export const typeOptions: {
  value: AnswerTableColumn["type"];
  label: ReactNode;
  icon: TablerIcon;
}[] = [
  { value: "str", label: "Text", icon: IconList },
  { value: "str_array", label: "List of text", icon: IconList },
  { value: "int", label: "Integer", icon: IconHash },
  { value: "number", label: "Number (float)", icon: IconHash },
  { value: "int_array", label: "List of integers", icon: IconHash },
  { value: "number_array", label: "List of numbers (float)", icon: IconHash },
  { value: "bool", label: "True / False", icon: IconCheck }
];

export const generateOptions: {
  value: boolean;
  label: ReactNode;
  icon: TablerIcon;
}[] = [
  { value: true, label: "Enabled", icon: IconAccessPoint },
  { value: false, label: "Disabled", icon: IconAccessPointOff }
];
