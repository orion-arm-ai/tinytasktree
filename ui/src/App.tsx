import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
    Background,
    Controls,
    Edge,
    Handle,
    MiniMap,
    Node,
    NodeProps,
    Position,
    ReactFlowProvider,
    ConnectionMode,
    BackgroundVariant,
} from "reactflow";
import {
    Alert,
    Button,
    Card,
    ConfigProvider,
    Descriptions,
    Input,
    Layout,
    Space,
    Switch,
    Table,
    Tabs,
    Tag,
    theme,
    Typography,
} from "antd";
import Highlighter from "react-highlight-words";
import "reactflow/dist/style.css";

const { Header, Content } = Layout;
const { Text, Title } = Typography;

const X_SPACING = 420;
const Y_SPACING = 190;
const MIN_LEFT_WIDTH = 360;
const MAX_LEFT_WIDTH = 1400;
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim().replace(/\/+$/, "") || "";

type KindIconProps = {
    className?: string;
};

type LucideNode = [tag: "path" | "circle" | "line" | "rect", attrs: Record<string, string>];

function LucideIcon({ nodes, className }: { nodes: LucideNode[]; className?: string }) {
    return (
        <svg
            className={className}
            xmlns="http://www.w3.org/2000/svg"
            width="1em"
            height="1em"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
        >
            {nodes.map(([tag, attrs], index) => React.createElement(tag, { key: index, ...attrs }))}
        </svg>
    );
}

const TreesIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["path", { d: "M10 10v.2A3 3 0 0 1 8.9 16H5a3 3 0 0 1-1-5.8V10a3 3 0 0 1 6 0Z" }],
            ["path", { d: "M7 16v6" }],
            ["path", { d: "M13 19v3" }],
            ["path", { d: "M12 19h8.3a1 1 0 0 0 .7-1.7L18 14h.3a1 1 0 0 0 .7-1.7L16 9h.2a1 1 0 0 0 .8-1.7L13 3l-1.4 1.5" }],
        ]}
    />
);

const WaypointsIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["circle", { cx: "12", cy: "4.5", r: "2.5" }],
            ["path", { d: "m10.2 6.3-3.9 3.9" }],
            ["circle", { cx: "4.5", cy: "12", r: "2.5" }],
            ["path", { d: "M7 12h10" }],
            ["circle", { cx: "19.5", cy: "12", r: "2.5" }],
            ["path", { d: "m13.8 17.7 3.9-3.9" }],
            ["circle", { cx: "12", cy: "19.5", r: "2.5" }],
        ]}
    />
);

const GitBranchIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["line", { x1: "6", x2: "6", y1: "3", y2: "15" }],
            ["circle", { cx: "18", cy: "6", r: "3" }],
            ["circle", { cx: "6", cy: "18", r: "3" }],
            ["path", { d: "M18 9a9 9 0 0 1-9 9" }],
        ]}
    />
);

const SplitIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["path", { d: "M16 3h5v5" }],
            ["path", { d: "M8 3H3v5" }],
            ["path", { d: "M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3" }],
            ["path", { d: "m15 9 6-6" }],
        ]}
    />
);

const NetworkIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["rect", { x: "16", y: "16", width: "6", height: "6", rx: "1" }],
            ["rect", { x: "2", y: "16", width: "6", height: "6", rx: "1" }],
            ["rect", { x: "9", y: "2", width: "6", height: "6", rx: "1" }],
            ["path", { d: "M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3" }],
            ["path", { d: "M12 12V8" }],
        ]}
    />
);

const BracesIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["path", { d: "M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1" }],
            ["path", { d: "M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1" }],
        ]}
    />
);

const ScaleIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["path", { d: "m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" }],
            ["path", { d: "m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z" }],
            ["path", { d: "M7 21h10" }],
            ["path", { d: "M12 3v18" }],
            ["path", { d: "M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2" }],
        ]}
    />
);

const WandSparklesIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["path", { d: "m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72" }],
            ["path", { d: "m14 7 3 3" }],
            ["path", { d: "M5 6v4" }],
            ["path", { d: "M19 14v4" }],
            ["path", { d: "M10 2v2" }],
            ["path", { d: "M7 8H3" }],
            ["path", { d: "M21 16h-4" }],
            ["path", { d: "M11 3H9" }],
        ]}
    />
);

const GitCommitHorizontalIcon = ({ className }: KindIconProps) => (
    <LucideIcon
        className={className}
        nodes={[
            ["circle", { cx: "12", cy: "12", r: "3" }],
            ["line", { x1: "3", x2: "9", y1: "12", y2: "12" }],
            ["line", { x1: "15", x2: "21", y1: "12", y2: "12" }],
        ]}
    />
);

type TraceNodeJson = {
    name: string;
    kind: string;
    start_at: string;
    end_at: string;
    duration: number;
    finished: boolean;
    cost: number;
    logs: string[];
    result: string | null;
    attributes: Record<string, string>;
    children: Record<string, TraceNodeJson>;
    total_tokens?: TokenUsage;
};

type FlattenedNode = {
    id: string;
    parentId: string | null;
    depth: number;
    order: number;
    node: TraceNodeJson;
};

type StackRow = {
    id: string;
    parentId: string | null;
    depth: number;
    order: number;
    node: TraceNodeJson;
    childIds: string[];
    status: TraceStatus;
    subtreeColor: string | null;
};

type StackIndex = {
    rows: StackRow[];
    rowMap: Map<string, StackRow>;
};

type NodeTone = {
    label: string;
    className: string;
    compactClassName: string;
    accent: string;
    icon: React.ComponentType<{ className?: string }>;
};

type TraceStatus = "OK" | "FAIL" | "UNKNOWN";

type TraceNodeData = {
    node: TraceNodeJson;
    status: TraceStatus;
    durationMs: number;
    cost: number;
    subtreeColor: string | null;
    isFolded: boolean;
    isSelected?: boolean;
};

type TokenUsage = {
    prompt?: number;
    completion?: number;
    total?: number;
};

type TraceListItem = {
    id: string;
    name: string;
    created_at: string;
};

const SUBTREE_COLORS = [
    "#ea580c",
    "#0ea5e9",
    "#65a30d",
    "#4f46e5",
    "#f43f5e",
    "#0891b2",
    "#059669",
    "#c026d3",
    "#2563eb",
    "#ec4899",
    "#3f6212",
    "#7c3aed",
    "#334155",
    "#92400e",
    "#14b8a6",
];

function hexToRgba(hex: string, alpha: number): string {
    const clean = hex.replace("#", "");
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function toPastel(hex: string | null | undefined, alpha = 0.25): string {
    if (!hex) return "rgba(170,170,170,0.28)";
    return hexToRgba(hex, alpha);
}

function parseStatus(resultStr: unknown): TraceStatus {
    if (!resultStr) {
        return "UNKNOWN";
    }
    if (typeof resultStr === "object" && resultStr !== null && "status" in resultStr) {
        const statusValue = (resultStr as { status?: number | string }).status;
        if (statusValue === 0 || statusValue === "OK") return "OK";
        if (statusValue === 1 || statusValue === "FAIL") return "FAIL";
    }
    if (typeof resultStr !== "string") {
        return "UNKNOWN";
    }
    try {
        const parsed = JSON.parse(resultStr);
        if (parsed && typeof parsed.status !== "undefined") {
            return parsed.status === 0 ? "OK" : "FAIL";
        }
    } catch {
        // ignore
    }
    if (resultStr.includes("OK(")) return "OK";
    if (resultStr.includes("FAIL(")) return "FAIL";
    return "UNKNOWN";
}

function parseTokenUsage(raw: unknown): TokenUsage | null {
    if (raw == null) return null;
    const normalized = () => {
        if (typeof raw === "string") {
            const trimmed = raw.trim();
            if (!trimmed || trimmed === "None") return null;
            try {
                return JSON.parse(trimmed) as unknown;
            } catch {
                return null;
            }
        }
        return raw;
    };
    const value = normalized();
    if (!value || typeof value !== "object") return null;
    const obj = value as Record<string, unknown>;
    const toNum = (v: unknown): number | null => {
        if (v == null) return null;
        const num = Number(v);
        return Number.isFinite(num) ? Math.round(num) : null;
    };
    const prompt = toNum(obj.prompt ?? obj.prompt_tokens);
    const completion = toNum(obj.completion ?? obj.completion_tokens);
    const computedTotal = (prompt ?? 0) + (completion ?? 0);
    const total = toNum(obj.total ?? obj.total_tokens ?? computedTotal);
    if (prompt == null && completion == null && total == null) return null;
    return { prompt: prompt ?? undefined, completion: completion ?? undefined, total: total ?? undefined };
}

function formatTokenUsage(usage: TokenUsage | null): string {
    if (!usage) return "";
    const total = usage.total ?? (usage.prompt ?? 0) + (usage.completion ?? 0);
    if (usage.prompt != null || usage.completion != null) {
        const prompt = usage.prompt ?? 0;
        const completion = usage.completion ?? 0;
        return `Tokens: ${total} (${prompt}p/${completion}c)`;
    }
    if (usage.total != null) return `Tokens: ${usage.total}`;
    return "";
}

function tokenUsageFromAttributes(attrs: Record<string, string> | null | undefined): TokenUsage | null {
    if (!attrs) return null;
    const fromTokens = parseTokenUsage(attrs.tokens);
    if (fromTokens) return fromTokens;
    const toNum = (value: unknown): number | null => {
        if (value == null) return null;
        const num = Number(value);
        return Number.isFinite(num) ? Math.round(num) : null;
    };
    const prompt = toNum(attrs.prompt_tokens ?? attrs.prompt);
    const completion = toNum(attrs.completion_tokens ?? attrs.completion);
    const total = toNum(attrs.total_tokens ?? attrs.total);
    if (prompt == null && completion == null && total == null) return null;
    return {
        prompt: prompt ?? undefined,
        completion: completion ?? undefined,
        total: total ?? undefined,
    };
}

function tokenizeQuery(query: string): string[] {
    return query
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);
}

function escapeRegex(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function apiUrl(path: string): string {
    if (!API_BASE_URL) return path;
    return `${API_BASE_URL}${path}`;
}

function parseDate(value: string): number {
    const ts = Date.parse(value);
    return Number.isNaN(ts) ? 0 : ts;
}

function sortChildren(children: Record<string, TraceNodeJson>): TraceNodeJson[] {
    return Object.values(children).sort((a, b) => parseDate(a.start_at) - parseDate(b.start_at));
}

function hasResultContent(raw: unknown): boolean {
    if (raw == null) return false;
    if (typeof raw === "object") {
        const obj = raw as { data?: unknown };
        if ("data" in obj) {
            const data = obj.data;
            if (data == null) return false;
            if (typeof data === "string" && data.trim() === "None") return false;
            return true;
        }
        return true;
    }
    if (typeof raw === "string") {
        const trimmed = raw.trim();
        if (!trimmed || trimmed === "None") return false;
        try {
            const parsed = JSON.parse(trimmed);
            if (parsed && typeof parsed === "object" && "data" in parsed) {
                const data = (parsed as { data?: unknown }).data;
                if (data == null) return false;
                if (typeof data === "string" && data.trim() === "None") return false;
                return true;
            }
        } catch {
            // ignore
        }
        return true;
    }
    return true;
}

function extractResultData(raw: unknown): string {
    if (raw == null) return "";
    if (typeof raw === "object") {
        const obj = raw as { data?: unknown };
        if ("data" in obj) {
            const data = obj.data;
            if (data == null) return "";
            if (typeof data === "string") return data;
            return JSON.stringify(data, null, 2);
        }
        return JSON.stringify(raw, null, 2);
    }
    if (typeof raw === "string") {
        const trimmed = raw.trim();
        if (!trimmed) return "";
        try {
            const parsed = JSON.parse(trimmed);
            if (parsed && typeof parsed === "object" && "data" in parsed) {
                const data = (parsed as { data?: unknown }).data;
                if (data == null) return "";
                if (typeof data === "string") return data;
                return JSON.stringify(data, null, 2);
            }
        } catch {
            // ignore
        }
        return trimmed;
    }
    return String(raw);
}

function shouldFoldChildren(node: TraceNodeJson): boolean {
    const raw = node.attributes?.suggest_fold_children;
    if (!raw) return false;
    if (typeof raw === "boolean") return raw;
    if (typeof raw === "number") return raw !== 0;
    if (typeof raw === "string") {
        const text = raw.trim().toLowerCase();
        if (text === "true" || text === "1" || text === "yes") return true;
        try {
            const parsed = JSON.parse(raw);
            if (typeof parsed === "boolean") return parsed;
            if (typeof parsed === "number") return parsed !== 0;
            if (typeof parsed === "string") return ["true", "1", "yes"].includes(parsed.toLowerCase());
        } catch {
            return false;
        }
    }
    return false;
}

function buildLayout(root: TraceNodeJson, maxDepth: number | null, activeFoldIds: Set<string>) {
    const out: FlattenedNode[] = [];
    let index = 0;
    let yCursor = 0;
    const pos = new Map<string, { x: number; y: number }>();

    const walk = (node: TraceNodeJson, parentId: string | null, depth: number, pathId: string): string => {
        const id = pathId;
        const foldByAttr = activeFoldIds.has(id);
        const depthLimited = maxDepth !== null && depth >= maxDepth;
        if (foldByAttr || depthLimited) {
            const childCount = Object.keys(node.children || {}).length;
            const nameSuffix = childCount ? ` (folded ${childCount})` : " (folded)";
            out.push({
                id,
                parentId,
                depth,
                order: index++,
                node: { ...node, name: `${node.name || ""}${nameSuffix}` },
            });
            const y = yCursor;
            yCursor += Y_SPACING;
            pos.set(id, { x: depth * X_SPACING, y });
            return id;
        }
        out.push({ id, parentId, depth, order: index++, node });
        const kids = sortChildren(node.children || {});
        if (!kids.length) {
            const y = yCursor;
            yCursor += Y_SPACING;
            pos.set(id, { x: depth * X_SPACING, y });
            return id;
        }
        let firstChildY = 0;
        kids.forEach((child, idx) => {
            const childId = walk(child, id, depth + 1, `${id}.${idx}`);
            const childPos = pos.get(childId);
            if (idx === 0 && childPos) firstChildY = childPos.y;
        });
        pos.set(id, { x: depth * X_SPACING, y: firstChildY });
        return id;
    };

    walk(root, null, 0, "root");
    return { flat: out, pos };
}

function buildFlatAll(root: TraceNodeJson) {
    const out: FlattenedNode[] = [];
    let index = 0;

    const walk = (node: TraceNodeJson, parentId: string | null, depth: number, pathId: string) => {
        out.push({ id: pathId, parentId, depth, order: index++, node });
        const kids = sortChildren(node.children || {});
        kids.forEach((child, idx) => walk(child, pathId, depth + 1, `${pathId}.${idx}`));
    };

    walk(root, null, 0, "root");
    return out;
}

function buildSubtreeColorMap(flat: FlattenedNode[]) {
    const childrenByParent = new Map<string | null, string[]>();
    flat.forEach((item) => {
        const list = childrenByParent.get(item.parentId) || [];
        list.push(item.id);
        childrenByParent.set(item.parentId, list);
    });

    const depthMap = new Map<string, number>();
    flat.forEach((item) => depthMap.set(item.id, item.depth));

    const subtreeColorMap = new Map<string, string | null>();
    const subtreeRoots = flat
        .filter((item) => item.node.kind === "Tree")
        .map((item) => item.id)
        .sort((a, b) => (depthMap.get(b) || 0) - (depthMap.get(a) || 0));

    const assignSubtree = (rootId: string, color: string) => {
        const stack = [rootId];
        while (stack.length) {
            const id = stack.pop() as string;
            if (!subtreeColorMap.has(id)) {
                subtreeColorMap.set(id, color);
            }
            const kids = childrenByParent.get(id) || [];
            kids.forEach((kid) => {
                if (!subtreeColorMap.has(kid)) stack.push(kid);
            });
        }
    };

    subtreeRoots.forEach((rootId, index) => {
        assignSubtree(rootId, SUBTREE_COLORS[index % SUBTREE_COLORS.length]);
    });

    return subtreeColorMap;
}

function buildStackIndex(root: TraceNodeJson): StackIndex {
    const flat = buildFlatAll(root);
    const subtreeColorMap = buildSubtreeColorMap(flat);
    const rows = flat.map((item) => ({
        id: item.id,
        parentId: item.parentId,
        depth: item.depth,
        order: item.order,
        node: item.node,
        childIds: flat.filter((candidate) => candidate.parentId === item.id).map((candidate) => candidate.id),
        status: parseStatus(item.node.result || null),
        subtreeColor: subtreeColorMap.get(item.id) || null,
    }));
    return {
        rows,
        rowMap: new Map(rows.map((row) => [row.id, row])),
    };
}

function buildVisibleStackRows(index: StackIndex, activeFoldIds: Set<string>) {
    const out: StackRow[] = [];
    const visit = (id: string) => {
        const row = index.rowMap.get(id);
        if (!row) return;
        out.push(row);
        if (!row.childIds.length || activeFoldIds.has(id)) return;
        row.childIds.forEach(visit);
    };
    if (index.rowMap.has("root")) visit("root");
    return out;
}

function buildPathIds(id: string | null | undefined): string[] {
    if (!id) return [];
    const parts = id.split(".");
    return parts.map((_, index) => parts.slice(0, index + 1).join("."));
}

function totalCost(node: TraceNodeJson): number {
    let sum = node.cost || 0;
    for (const child of Object.values(node.children || {})) {
        sum += totalCost(child);
    }
    return sum;
}

function formatCost(cost: number): string {
    return `$${cost.toFixed(6)}`;
}

function formatTraceListDate(value: string): string {
    const ts = Date.parse(value);
    if (Number.isNaN(ts)) return value;
    return new Date(ts).toLocaleString();
}

function formatStackDuration(value: number): string {
    if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)}s`;
    if (value >= 100) return `${value.toFixed(0)}ms`;
    if (value <= 0) return "0ms";
    return `${value.toFixed(1)}ms`;
}

function normalizeNodeKind(kind: string | undefined): string {
    switch (kind || "") {
        case "Function":
            return "Lambda";
        case "Log":
        case "TODO":
        case "Failure":
        case "WriteBlackboard":
        case "ShowBlackboard":
        case "Subtree":
        case "ParseJSON":
            return "Leaf";
        case "Assertion":
        case "If":
        case "Else":
            return "Condition";
        case "ForceOk":
        case "ForceFail":
        case "Invert":
        case "Return":
        case "Retry":
        case "While":
        case "Timeout":
        case "Fallback":
        case "Terminable":
        case "RedisCacher":
        case "Wrapper":
            return "Decorator";
        case "RandomSelector":
            return "Selector";
        default:
            return kind || "Node";
    }
}

function getNodeTone(kind: string | undefined): NodeTone {
    const normalizedKind = normalizeNodeKind(kind);
    switch (normalizedKind) {
        case "Tree":
            return {
                label: kind || "Tree",
                className: "stack-kind-tone tree",
                compactClassName: "stack-kind-compact tree",
                accent: "#0f7a3c",
                icon: TreesIcon,
            };
        case "Sequence":
            return {
                label: kind || "Sequence",
                className: "stack-kind-tone sequence",
                compactClassName: "stack-kind-compact sequence",
                accent: "#0ea5e9",
                icon: WaypointsIcon,
            };
        case "Selector":
            return {
                label: kind || "Selector",
                className: "stack-kind-tone selector",
                compactClassName: "stack-kind-compact selector",
                accent: "#7c3aed",
                icon: GitBranchIcon,
            };
        case "Parallel":
            return {
                label: kind || "Parallel",
                className: "stack-kind-tone parallel",
                compactClassName: "stack-kind-compact parallel",
                accent: "#d97706",
                icon: SplitIcon,
            };
        case "Gather":
            return {
                label: kind || "Gather",
                className: "stack-kind-tone gather",
                compactClassName: "stack-kind-compact gather",
                accent: "#0891b2",
                icon: NetworkIcon,
            };
        case "LLM":
            return {
                label: kind || "LLM",
                className: "stack-kind-tone llm",
                compactClassName: "stack-kind-compact llm",
                accent: "#2563eb",
                icon: WandSparklesIcon,
            };
        case "Lambda":
            return {
                label: kind || "Lambda",
                className: "stack-kind-tone default",
                compactClassName: "stack-kind-compact default",
                accent: "#d946ef",
                icon: BracesIcon,
            };
        case "Condition":
            return {
                label: kind || "Condition",
                className: "stack-kind-tone default",
                compactClassName: "stack-kind-compact default",
                accent: "#84cc16",
                icon: ScaleIcon,
            };
        case "Decorator":
            return {
                label: kind || "Decorator",
                className: "stack-kind-tone default",
                compactClassName: "stack-kind-compact default",
                accent: "#f97316",
                icon: WandSparklesIcon,
            };
        case "Leaf":
            return {
                label: kind || "Leaf",
                className: "stack-kind-tone default",
                compactClassName: "stack-kind-compact default",
                accent: "#94a3b8",
                icon: GitCommitHorizontalIcon,
            };
        default:
            return {
                label: kind || "Node",
                className: "stack-kind-tone default",
                compactClassName: "stack-kind-compact default",
                accent: "#64748b",
                icon: GitCommitHorizontalIcon,
            };
    }
}

function statusColor(status: TraceStatus): string {
    if (status === "OK") return "green";
    if (status === "FAIL") return "red";
    return "default";
}

function TraceCard(props: NodeProps<TraceNodeData>) {
    const { data, selected } = props;
    const isSelected = data.isSelected || selected;
    const status = data.status;
    const tone = getNodeTone(data.node.kind);
    const KindIcon = tone.icon;
    let borderColor = "#3fb950";
    if (status === "FAIL") borderColor = "#f85149";
    if (status === "UNKNOWN") borderColor = "#6e7681";
    if (isSelected) borderColor = "#58a6ff";
    const borderWidth = isSelected ? 6 : status === "FAIL" ? 5 : 4;
    const background = data.subtreeColor
        ? `linear-gradient(135deg, rgba(13, 17, 23, 0.98) 0%, ${hexToRgba(data.subtreeColor, 0.18)} 100%)`
        : "linear-gradient(180deg, rgba(17, 24, 39, 0.98) 0%, rgba(13, 17, 23, 0.98) 100%)";
    const foldedPrefix = data.isFolded ? "▸ " : "";
    const tokenUsage = data.node.kind === "LLM" ? tokenUsageFromAttributes(data.node.attributes) : null;
    const tokenLine = formatTokenUsage(tokenUsage);

    return (
        <div
            className="trace-card"
            style={{
                borderColor,
                borderWidth,
                background,
                borderStyle: data.isFolded ? "dashed" : "solid",
            }}
        >
            <Handle type="target" position={Position.Left} className="trace-handle" />
            <Handle type="source" position={Position.Right} className="trace-handle" />
            <div className="trace-card-title">
                <span className="trace-card-kind-icon" style={{ "--tone-accent": tone.accent } as React.CSSProperties}>
                    <KindIcon />
                </span>
                <span className="trace-card-name">
                    {foldedPrefix}
                    {data.node.name || "(unnamed)"}
                </span>
            </div>
            <Tag color={statusColor(status)} className="trace-status">
                {status}
            </Tag>
            <div className="trace-card-meta">Duration: {data.durationMs.toFixed(2)} ms</div>
            <div className="trace-card-meta">Cost: {formatCost(data.cost)}</div>
            {tokenLine && <div className="trace-card-meta">{tokenLine}</div>}
        </div>
    );
}

const nodeTypes = { traceNode: TraceCard };

function useResizableSidebar(initialWidth: number) {
    const [width, setWidth] = useState(initialWidth);
    const dragging = useRef(false);

    useEffect(() => {
        const onMove = (event: MouseEvent) => {
            if (!dragging.current) return;
            const next = Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, event.clientX));
            setWidth(next);
        };
        const onUp = () => {
            dragging.current = false;
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
        return () => {
            window.removeEventListener("mousemove", onMove);
            window.removeEventListener("mouseup", onUp);
        };
    }, []);

    const onMouseDown = useCallback(() => {
        dragging.current = true;
    }, []);

    return { width, onMouseDown };
}

function copyText(text: string) {
    void navigator.clipboard.writeText(text);
}

function TraceUI() {
    const [trace, setTrace] = useState<TraceNodeJson | null>(null);
    const [traceList, setTraceList] = useState<TraceListItem[]>([]);
    const [traceListLoading, setTraceListLoading] = useState(false);
    const [viewMode, setViewMode] = useState<"flow" | "stack">("stack");
    const [compactMode, setCompactMode] = useState(true);
    const [stackOrderMode, setStackOrderMode] = useState<"tree" | "time" | "cost" | "error">("tree");
    const [stackLeafOnly, setStackLeafOnly] = useState(false);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [tab, setTab] = useState<"result" | "logs" | "attributes">("result");
    const initialLeftWidth = useMemo(() => {
        if (typeof window === "undefined") return 640;
        return Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, Math.floor(window.innerWidth * 0.65)));
    }, []);
    const resizer = useResizableSidebar(initialLeftWidth);
    const [searchTerm, setSearchTerm] = useState("");
    const [searchOpen, setSearchOpen] = useState(false);
    const [searchActiveIndex, setSearchActiveIndex] = useState(0);
    const [flowInstance, setFlowInstance] = useState<any>(null);
    const searchListRef = useRef<HTMLDivElement | null>(null);
    const stackListRef = useRef<HTMLDivElement | null>(null);
    const fitTimer = useRef<number | null>(null);
    const maxDepth: number | null = null;
    const [autoFoldOn, setAutoFoldOn] = useState(true);
    const [manualFoldIds, setManualFoldIds] = useState<Set<string>>(new Set());
    const [manualExpandIds, setManualExpandIds] = useState<Set<string>>(new Set());
    const [suggestFoldIds, setSuggestFoldIds] = useState<Set<string>>(new Set());

    useEffect(() => {
        if (!trace) return;
        const ids = new Set<string>();
        const walk = (node: TraceNodeJson, pathId: string) => {
            if (shouldFoldChildren(node)) ids.add(pathId);
            const kids = sortChildren(node.children || {});
            kids.forEach((child, idx) => walk(child, `${pathId}.${idx}`));
        };
        walk(trace, "root");
        setSuggestFoldIds(ids);
        setManualFoldIds(new Set());
        setManualExpandIds(new Set());
    }, [trace]);

    const activeFoldIds = useMemo(() => {
        const result = new Set<string>();
        manualFoldIds.forEach((id) => result.add(id));
        if (autoFoldOn) {
            suggestFoldIds.forEach((id) => {
                if (!manualExpandIds.has(id)) result.add(id);
            });
        }
        return result;
    }, [manualFoldIds, manualExpandIds, suggestFoldIds, autoFoldOn]);

    const fuzzyMatch = useCallback((text: string, query: string) => {
        const hay = text.toLowerCase();
        const tokens = tokenizeQuery(query);
        if (!tokens.length) return null;
        const hits: number[] = [];
        for (const token of tokens) {
            const idx = hay.indexOf(token);
            if (idx === -1) return null;
            hits.push(idx);
        }
        return hits.length ? hits : null;
    }, []);

    useEffect(() => {
        const pathId = window.location.pathname.replace(/^\/+/, "");
        if (pathId) {
            void loadTrace(pathId);
            return;
        }
        void loadTraceList();
    }, []);

    const loadTraceList = useCallback(async () => {
        setTraceListLoading(true);
        setLoadError(null);
        setTrace(null);
        setSelectedId(null);
        try {
            const resp = await fetch(apiUrl("/traces"));
            if (!resp.ok) {
                let detail = `Failed to load traces: ${resp.status}`;
                try {
                    const errBody = await resp.json();
                    if (typeof errBody?.error === "string" && errBody.error) detail = errBody.error;
                    else if (typeof errBody?.detail === "string" && errBody.detail) detail = errBody.detail;
                } catch {
                    // ignore
                }
                throw new Error(detail);
            }
            const data = (await resp.json()) as TraceListItem[];
            setTraceList(Array.isArray(data) ? data : []);
            if (window.location.pathname !== "/") {
                window.history.replaceState({}, "", "/");
            }
        } catch (err) {
            setLoadError(err instanceof Error ? err.message : "Unknown error");
            setTraceList([]);
        } finally {
            setTraceListLoading(false);
        }
    }, []);

    const loadTrace = useCallback(async (id: string) => {
        setLoadError(null);
        setTrace(null);
        setTraceList([]);
        setSelectedId(null);
        try {
            const resp = await fetch(apiUrl(`/trace/${encodeURIComponent(id)}`));
            if (!resp.ok) {
                let detail = `Failed to load trace: ${resp.status}`;
                try {
                    const errBody = await resp.json();
                    if (typeof errBody?.error === "string" && errBody.error) detail = errBody.error;
                    else if (typeof errBody?.detail === "string" && errBody.detail) detail = errBody.detail;
                } catch {
                    // ignore
                }
                throw new Error(detail);
            }
            const data = (await resp.json()) as TraceNodeJson;
            setTrace(data);
            setSelectedId(null);
            if (window.location.pathname !== `/${id}`) {
                window.history.replaceState({}, "", `/${id}`);
            }
        } catch (err) {
            setLoadError(err instanceof Error ? err.message : "Unknown error");
        }
    }, []);

    const { nodes, edges, nodeMap } = useMemo(() => {
        if (!trace) {
            return {
                nodes: [] as Node<TraceNodeData>[],
                edges: [] as Edge[],
                nodeMap: new Map<string, FlattenedNode>(),
            };
        }
        const { flat, pos } = buildLayout(trace, maxDepth, activeFoldIds);
        const nodeMapLocal = new Map<string, FlattenedNode>();
        flat.forEach((item) => nodeMapLocal.set(item.id, item));
        const subtreeColorMap = buildSubtreeColorMap(flat);

        const resolveSubtreeColor = (id: string): string | null => subtreeColorMap.get(id) || null;
        const nodesLocal: Node<TraceNodeData>[] = flat.map((item) => {
            const status = parseStatus(item.node.result || null);
            const position = pos.get(item.id) || { x: item.depth * X_SPACING, y: item.order * Y_SPACING };
            const subtreeColor = resolveSubtreeColor(item.id);
            const isFolded = activeFoldIds.has(item.id);
            return {
                id: item.id,
                type: "traceNode",
                data: {
                    node: item.node,
                    status,
                    durationMs: item.node.duration || 0,
                    cost: item.node.cost || 0,
                    subtreeColor,
                    isFolded,
                    isSelected: selectedId === item.id,
                },
                position: {
                    x: position.x,
                    y: position.y,
                },
            };
        });
        const edgesLocal: Edge[] = flat
            .filter((item) => item.parentId)
            .map((item) => ({
                id: `e_${item.parentId}_${item.id}`,
                source: item.parentId as string,
                target: item.id,
                type: "smoothstep",
                markerEnd: undefined,
                markerStart: undefined,
            }));
        return { nodes: nodesLocal, edges: edgesLocal, nodeMap: nodeMapLocal };
    }, [trace, maxDepth, autoFoldOn, activeFoldIds, selectedId]);

    const { searchIndex, searchNodeMap } = useMemo(() => {
        if (!trace) {
            return {
                searchIndex: [] as { id: string; text: string }[],
                searchNodeMap: new Map<string, FlattenedNode>(),
            };
        }
        const flatAll = buildFlatAll(trace);
        const nodeMapLocal = new Map<string, FlattenedNode>();
        flatAll.forEach((item) => nodeMapLocal.set(item.id, item));
        const searchIndexLocal = flatAll.map((item) => {
            const blob = JSON.stringify({
                name: item.node.name,
                kind: item.node.kind,
                result: item.node.result,
                logs: item.node.logs,
                attributes: item.node.attributes,
            });
            return { id: item.id, text: blob };
        });
        return { searchIndex: searchIndexLocal, searchNodeMap: nodeMapLocal };
    }, [trace]);

    const stackIndex = useMemo(() => {
        if (!trace) return null;
        return buildStackIndex(trace);
    }, [trace]);

    const visibleStackRows = useMemo(() => {
        if (!stackIndex) return [] as StackRow[];
        if (stackOrderMode === "tree") return buildVisibleStackRows(stackIndex, activeFoldIds);

        const rows = stackIndex.rows.filter((row) => {
            if (row.id === "root") return false;
            if (stackOrderMode === "error" && row.status !== "FAIL") return false;
            if (stackLeafOnly && row.childIds.length > 0) return false;
            return true;
        });

        rows.sort((a, b) => {
            const primary =
                stackOrderMode === "time"
                    ? (b.node.duration || 0) - (a.node.duration || 0)
                    : stackOrderMode === "cost"
                      ? (b.node.cost || 0) - (a.node.cost || 0)
                      : (b.node.duration || 0) - (a.node.duration || 0);
            if (primary !== 0) return primary;
            return b.depth - a.depth;
        });

        return rows;
    }, [stackIndex, activeFoldIds, stackOrderMode, stackLeafOnly]);
    const isTreeStackMode = stackOrderMode === "tree";

    const total = trace ? totalCost(trace) : 0;

    const stackStats = useMemo(() => {
        const rows = stackIndex?.rows || [];
        return {
            nodeCount: rows.length,
            visibleCount: visibleStackRows.length,
            totalDuration: trace?.duration || 0,
            totalCost: total,
            maxDuration: Math.max(...rows.map((row) => row.node.duration || 0), 0),
            maxCost: Math.max(...rows.map((row) => row.node.cost || 0), 0),
        };
    }, [stackIndex, visibleStackRows.length, trace, total]);

    const selectedNode = selectedId ? searchNodeMap.get(selectedId) || nodeMap.get(selectedId) : null;
    const selectedTrace = selectedNode?.node || null;
    const selectedPathIds = useMemo(() => buildPathIds(selectedId), [selectedId]);
    const resultText = useMemo(() => {
        if (!selectedTrace?.result) return "";
        return extractResultData(selectedTrace.result);
    }, [selectedTrace]);

    useEffect(() => {
        if (!flowInstance || nodes.length === 0) return;
        if (fitTimer.current) {
            window.clearTimeout(fitTimer.current);
        }
        fitTimer.current = window.setTimeout(() => {
            flowInstance.fitView({ padding: 0.2, duration: 300 });
        }, 50);
        return () => {
            if (fitTimer.current) {
                window.clearTimeout(fitTimer.current);
            }
        };
    }, [flowInstance, nodes.length, activeFoldIds]);


    const searchResults = useMemo(() => {
        if (searchTerm.trim().length < 1) return [];
        const results: {
            id: string;
            name: string;
            hits: number[];
            nameHits: number[] | null;
            attrHits: number[] | null;
            resultHits: number[] | null;
        }[] = [];
        for (const item of searchIndex) {
            const node = searchNodeMap.get(item.id)?.node;
            if (!node) continue;
            const nameText = node.name || "";
            const attrsText = JSON.stringify(node.attributes || {});
            const resultText = node.result ? JSON.stringify(node.result) : "";
            const nameHits = nameText ? fuzzyMatch(nameText, searchTerm.trim()) : null;
            const attrHits = attrsText ? fuzzyMatch(attrsText, searchTerm.trim()) : null;
            const resultHits = resultText ? fuzzyMatch(resultText, searchTerm.trim()) : null;
            const hits = nameHits || attrHits || resultHits;
            if (!hits) continue;
            results.push({
                id: item.id,
                name: node.name || "(unnamed)",
                hits,
                nameHits,
                attrHits,
                resultHits,
            });
        }
        return results
            .sort((a, b) => {
                const aName = a.nameHits ? 1 : 0;
                const bName = b.nameHits ? 1 : 0;
                if (aName !== bName) return bName - aName;
                const aAttr = a.attrHits ? 1 : 0;
                const bAttr = b.attrHits ? 1 : 0;
                if (aAttr !== bAttr) return bAttr - aAttr;
                const aRes = a.resultHits ? 1 : 0;
                const bRes = b.resultHits ? 1 : 0;
                if (aRes !== bRes) return bRes - aRes;
                return a.hits.length - b.hits.length;
            })
            .slice(0, 12);
    }, [searchIndex, searchTerm, searchNodeMap, fuzzyMatch]);

    const highlightText = useCallback((text: string, query: string) => {
        const tokens = tokenizeQuery(query);
        if (!tokens.length) return text;
        return (
            <Highlighter
                searchWords={tokens}
                textToHighlight={text}
                autoEscape
                highlightClassName="search-hit"
            />
        );
    }, []);

    const buildSnippet = useCallback((text: string, query: string, hits: number[]) => {
        if (!text) return "";
        const lower = text.toLowerCase();
        const q = query.toLowerCase();
        let start = 0;
        const direct = lower.indexOf(q);
        if (direct >= 0) {
            start = Math.max(0, direct - 40);
        } else if (hits.length > 0) {
            start = Math.max(0, hits[0] - 40);
        }
        const slice = text.slice(start, start + 160);
        return start > 0 ? `…${slice}` : slice;
    }, []);

    const highlightContent = useCallback((text: string) => highlightText(text, searchTerm.trim()), [highlightText, searchTerm]);

    const focusNodeById = useCallback(
        (id: string) => {
            if (!flowInstance) return;
            const live = flowInstance.getNode(id);
            if (live) {
                const w = live.width ?? 200;
                const h = live.height ?? 120;
                const cx = live.position.x + w / 2;
                const cy = live.position.y + h / 2;
                flowInstance.setCenter(cx, cy, { zoom: 1.2, duration: 350 });
                return;
            }
            const fallback = nodeMap.get(id);
            if (fallback) {
                flowInstance.setCenter(fallback.depth * X_SPACING + 120, fallback.order * Y_SPACING + 60, {
                    zoom: 1.2,
                    duration: 350,
                });
            }
        },
        [flowInstance, nodeMap]
    );

    useEffect(() => {
        if (viewMode !== "flow" || !selectedId || !flowInstance) return;
        const handle = window.setTimeout(() => {
            focusNodeById(selectedId);
        }, 80);
        return () => window.clearTimeout(handle);
    }, [selectedId, flowInstance, focusNodeById, nodes.length, activeFoldIds, viewMode]);

    useEffect(() => {
        if (viewMode !== "stack" || !selectedId || !stackListRef.current) return;
        const target = stackListRef.current.querySelector<HTMLElement>(`[data-stack-id="${selectedId}"]`);
        target?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }, [selectedId, viewMode, visibleStackRows.length]);

    const resetSearchDropdown = useCallback(() => {
        setSearchOpen(false);
        setSearchActiveIndex(0);
    }, []);

    const expandToNode = useCallback(
        (id: string) => {
            const parts = id.split(".");
            const ancestors: string[] = [];
            for (let i = 1; i <= parts.length; i += 1) {
                ancestors.push(parts.slice(0, i).join("."));
            }
            setManualFoldIds((prev) => {
                const next = new Set(prev);
                ancestors.forEach((a) => next.delete(a));
                return next;
            });
            setManualExpandIds((prev) => {
                const next = new Set(prev);
                ancestors.forEach((a) => next.add(a));
                return next;
            });
        },
        [setManualFoldIds, setManualExpandIds]
    );

    const toggleFoldById = useCallback(
        (id: string, hasChildren: boolean, depth: number) => {
            if (!hasChildren) return;
            const depthLimited = maxDepth !== null && depth >= maxDepth;
            if (depthLimited) return;
            const isSuggested = suggestFoldIds.has(id);
            const isManualFolded = manualFoldIds.has(id);
            const isExpandedOverride = manualExpandIds.has(id);
            const isFolded = isManualFolded || (autoFoldOn && isSuggested && !isExpandedOverride);

            if (isFolded) {
                setManualFoldIds((prev) => {
                    const next = new Set(prev);
                    next.delete(id);
                    return next;
                });
                if (isSuggested) {
                    setManualExpandIds((prev) => new Set(prev).add(id));
                }
            } else {
                setManualExpandIds((prev) => {
                    const next = new Set(prev);
                    next.delete(id);
                    return next;
                });
                setManualFoldIds((prev) => new Set(prev).add(id));
            }
        },
        [autoFoldOn, manualExpandIds, manualFoldIds, maxDepth, suggestFoldIds]
    );

    const searchCards = useMemo(() => {
        return searchResults.map((item) => {
            const node = searchNodeMap.get(item.id)?.node;
            const base = node
                ? JSON.stringify({
                      name: node.name,
                      attributes: node.attributes,
                      result: node.result,
                  })
                : "";
            const snippet = buildSnippet(base, searchTerm, item.hits);
            return {
                id: item.id,
                name: item.name || "(unnamed)",
                snippet,
            };
        });
    }, [searchResults, searchNodeMap, buildSnippet, searchTerm]);

    useEffect(() => {
        if (!searchOpen) return;
        if (searchActiveIndex >= searchCards.length) {
            setSearchActiveIndex(0);
        }
    }, [searchActiveIndex, searchCards.length, searchOpen]);

    const commitSearchSelection = useCallback(
        (item: { id: string; name: string }) => {
            setSearchTerm(item.name);
            expandToNode(item.id);
            setSelectedId(item.id);
            if (viewMode === "flow") {
                requestAnimationFrame(() => focusNodeById(item.id));
            }
            resetSearchDropdown();
        },
        [expandToNode, focusNodeById, resetSearchDropdown, viewMode]
    );

    const handleSearchKeyDown = useCallback(
        (event: React.KeyboardEvent<HTMLInputElement>) => {
            if (!searchOpen || searchCards.length === 0) return;
            if (event.key === "ArrowDown") {
                event.preventDefault();
                setSearchActiveIndex((prev) => Math.min(prev + 1, searchCards.length - 1));
                return;
            }
            if (event.key === "ArrowUp") {
                event.preventDefault();
                setSearchActiveIndex((prev) => Math.max(prev - 1, 0));
                return;
            }
            if (event.key === "Enter") {
                event.preventDefault();
                const target = searchCards[searchActiveIndex];
                if (target) commitSearchSelection(target);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                resetSearchDropdown();
            }
        },
        [searchOpen, searchCards, searchActiveIndex, commitSearchSelection, resetSearchDropdown]
    );

    useEffect(() => {
        if (!searchListRef.current) return;
        const active = searchListRef.current.querySelector<HTMLButtonElement>(
            ".search-item.active"
        );
        active?.scrollIntoView({ block: "nearest" });
    }, [searchActiveIndex]);

    const detailsItems = useMemo(() => {
        if (!selectedTrace) return [];
        const tokenUsage = selectedTrace.kind === "LLM" ? tokenUsageFromAttributes(selectedTrace.attributes) : null;
        const tokenLine = formatTokenUsage(tokenUsage);
        const items = [
            {
                key: "start",
                label: "Start",
                children: selectedTrace.start_at || "-",
            },
            {
                key: "end",
                label: "End",
                children: selectedTrace.end_at || "-",
            },
            {
                key: "duration",
                label: "Duration",
                children: `${selectedTrace.duration.toFixed(2)} ms`,
            },
        ];
        if (tokenLine) {
            items.push({
                key: "tokens",
                label: "Tokens",
                children: tokenLine.replace("Tokens: ", ""),
            });
        }
        if (selectedTrace.name === "ROOT") {
            const rootTokens = selectedTrace.total_tokens || null;
            const rootLine = formatTokenUsage(rootTokens);
            if (rootLine) {
                items.push({
                    key: "total_tokens",
                    label: "Total Tokens",
                    children: rootLine.replace("Tokens: ", ""),
                });
            }
        }
        if (selectedTrace.name === "ROOT") {
            items.push({
                key: "total",
                label: "Total Cost",
                children: formatCost(total),
            });
        }
        return items;
    }, [selectedTrace, total]);


    const detailsTabs = useMemo(
        () => [
            {
                key: "result",
                label: hasResultContent(selectedTrace?.result) ? "Result.data+" : "Result.data",
                children: (
                    <Card
                        size="small"
                        className="panel-section"
                        extra={
                            <Button size="small" onClick={() => copyText(resultText)}>
                                Copy
                            </Button>
                        }
                    >
                        <pre className="panel-content">{resultText ? highlightContent(resultText) : "(empty)"}</pre>
                    </Card>
                ),
            },
            {
                key: "logs",
                label: `Logs(${selectedTrace?.logs ? selectedTrace.logs.length : 0})`,
                children: (
                    <Card
                        size="small"
                        className="panel-section"
                        extra={
                            <Button size="small" onClick={() => copyText((selectedTrace?.logs || []).join("\n"))}>
                                Copy
                            </Button>
                        }
                    >
                        <pre className="panel-content">
                            {(selectedTrace?.logs || []).length
                                ? highlightContent((selectedTrace?.logs || []).join("\n"))
                                : "(empty)"}
                        </pre>
                    </Card>
                ),
            },
            {
                key: "attributes",
                label: `Attrs(${selectedTrace?.attributes ? Object.keys(selectedTrace.attributes).length : 0})`,
                children: (
                    <Card
                        size="small"
                        className="panel-section"
                        extra={
                            <Button
                                size="small"
                                onClick={() =>
                                    copyText(JSON.stringify(selectedTrace?.attributes || {}, null, 2))
                                }
                            >
                                Copy
                            </Button>
                        }
                    >
                        <Table
                            size="small"
                            pagination={false}
                            rowKey="key"
                            dataSource={Object.entries(selectedTrace?.attributes || {}).map(
                                ([key, value]) => ({ key, value })
                            )}
                            columns={[
                                {
                                    title: "Key",
                                    dataIndex: "key",
                                    key: "key",
                                    width: 140,
                                    render: (text: string) => <Text code>{text}</Text>,
                                },
                                {
                                    title: "Value",
                                    dataIndex: "value",
                                    key: "value",
                                    render: (value: unknown) => {
                                        const content =
                                            typeof value === "string"
                                                ? value
                                                : JSON.stringify(value, null, 2);
                                        return (
                                            <pre className="panel-content" style={{ margin: 0 }}>
                                                {content || "(empty)"}
                                            </pre>
                                        );
                                    },
                                },
                            ]}
                            scroll={{ x: 600 }}
                            className="attrs-table"
                            locale={{ emptyText: "(empty)" }}
                        />
                    </Card>
                ),
            },
        ],
        [selectedTrace, resultText, highlightContent]
    );

    return (
        <Layout className="app-shell">
            <Header className="top-bar">
                <Space size={16} align="center" className="top-bar-left">
                    <button type="button" className="brand-button" onClick={() => void loadTraceList()}>
                        <Title level={4} className="brand">
                            tinytasktree ui
                        </Title>
                    </button>
                </Space>
                <div className="search-box">
                    <Input
                        size="large"
                        placeholder="Search nodes..."
                        value={searchTerm}
                        onChange={(e) => {
                            setSearchTerm(e.target.value);
                            setSearchOpen(true);
                        }}
                        onKeyDown={handleSearchKeyDown}
                        onFocus={() => setSearchOpen(true)}
                        onBlur={() => setTimeout(() => resetSearchDropdown(), 150)}
                        className="search-input"
                    />
                    {searchOpen && searchCards.length > 0 && (
                        <div className="search-dropdown" ref={searchListRef}>
                            {searchCards.map((item, idx) => (
                                <button
                                    key={item.id}
                                    type="button"
                                    className={`search-item ${idx === searchActiveIndex ? "active" : ""}`}
                                    onMouseDown={(e) => e.preventDefault()}
                                    onClick={() => {
                                        commitSearchSelection(item);
                                    }}
                                >
                                    <div className="search-name">{highlightText(item.name, searchTerm)}</div>
                                    <div className="search-snippet">
                                        {item.snippet ? highlightText(item.snippet, searchTerm) : "-"}
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
                <Space size={16} align="center" className="top-controls">
                    <div className="view-mode-switch" role="tablist" aria-label="Trace view mode">
                        <button
                            type="button"
                            className={`view-mode-btn ${viewMode === "stack" ? "active" : ""}`}
                            onClick={() => setViewMode("stack")}
                        >
                            Stack
                        </button>
                        <button
                            type="button"
                            className={`view-mode-btn ${viewMode === "flow" ? "active" : ""}`}
                            onClick={() => setViewMode("flow")}
                        >
                            Flow
                        </button>
                    </div>
                    <Space direction="vertical" size={4} className="toggle-stack">
                        <Space align="center">
                            <Switch
                                checked={autoFoldOn}
                                onChange={(checked) => {
                                    setAutoFoldOn(checked);
                                    if (checked) {
                                        setSearchTerm("");
                                        setSearchOpen(false);
                                    }
                                }}
                                size="small"
                            />
                            <Text className="toggle-label">Auto fold</Text>
                        </Space>
                        <Text className="top-hint">Double-click a node to fold/unfold its subtree</Text>
                    </Space>
                </Space>
                {loadError && (
                    <Alert
                        type="error"
                        message={loadError}
                        showIcon
                        className="error-badge"
                    />
                )}
            </Header>

            <Content className="content">
                <div className="left-panel" style={{ width: resizer.width }}>
                    {trace ? (
                        viewMode === "flow" ? (
                            <ReactFlowProvider>
                                <ReactFlow
                                    nodes={nodes}
                                    edges={edges}
                                    nodeTypes={nodeTypes}
                                    style={{ width: "100%", height: "100%" }}
                                    onNodeClick={(_, node) => setSelectedId(node.id)}
                                    onNodeDoubleClick={(_, node) => {
                                        const target = nodeMap.get(node.id);
                                        if (!target) return;
                                        const hasChildren = Object.keys(target.node.children || {}).length > 0;
                                        toggleFoldById(node.id, hasChildren, target.depth);
                                    }}
                                    maxZoom={3}
                                    minZoom={0.05}
                                    zoomOnDoubleClick={false}
                                    connectionMode={ConnectionMode.Loose}
                                    fitView
                                    fitViewOptions={{ padding: 0.2, duration: 300 }}
                                    panOnScroll
                                    panOnScrollSpeed={0.7}
                                    onInit={(instance) => setFlowInstance(instance)}
                                >
                                    <Background gap={24} size={1} color="#AAAAAA" variant={BackgroundVariant.Dots} />
                                    <MiniMap
                                        pannable
                                        zoomable
                                        maskColor="rgba(0, 0, 0, 0.6)"
                                        style={{ backgroundColor: "#0d1117", border: "1px solid #30363d" }}
                                        nodeColor={(node) => {
                                            if ((node.data as TraceNodeData)?.isSelected || node.selected) return "#58a6ff";
                                            const data = node.data as TraceNodeData;
                                            if (data.status === "FAIL") return "#f85149";
                                            if (data.subtreeColor) {
                                                return data.subtreeColor;
                                            }
                                            return "#8b949e";
                                        }}
                                    />
                                    <Controls showInteractive={false} />
                                </ReactFlow>
                            </ReactFlowProvider>
                        ) : (
                            <div className="stack-view">
                                <div className="stack-summary">
                                    <div className="stack-stat">
                                        <span className="stack-stat-label">Nodes</span>
                                        <strong>{stackStats.nodeCount}</strong>
                                    </div>
                                    <div className="stack-stat">
                                        <span className="stack-stat-label">Visible</span>
                                        <strong>{stackStats.visibleCount}</strong>
                                    </div>
                                    <div className="stack-stat">
                                        <span className="stack-stat-label">Total Time</span>
                                        <strong className="stack-time-total">{formatStackDuration(stackStats.totalDuration)}</strong>
                                    </div>
                                    <div className="stack-stat">
                                        <span className="stack-stat-label">Total Cost</span>
                                        <strong className="stack-cost-total">{formatCost(stackStats.totalCost)}</strong>
                                    </div>
                                </div>
                                <div className="stack-toolbar">
                                    <div className="stack-actions">
                                        <Button size="small" type={compactMode ? "primary" : "default"} onClick={() => setCompactMode((v) => !v)}>
                                            {compactMode ? "Compact On" : "Compact Off"}
                                        </Button>
                                        <div className="stack-order-switch" role="tablist" aria-label="Stack sort mode">
                                            <button
                                                type="button"
                                                className={`stack-order-btn ${stackOrderMode === "tree" ? "active" : ""}`}
                                                onClick={() => setStackOrderMode("tree")}
                                            >
                                                Tree View
                                            </button>
                                            <button
                                                type="button"
                                                className={`stack-order-btn ${stackOrderMode === "time" ? "active" : ""}`}
                                                onClick={() => setStackOrderMode("time")}
                                            >
                                                Sort by Time
                                            </button>
                                            <button
                                                type="button"
                                                className={`stack-order-btn ${stackOrderMode === "cost" ? "active" : ""}`}
                                                onClick={() => setStackOrderMode("cost")}
                                            >
                                                Sort by Cost
                                            </button>
                                            <button
                                                type="button"
                                                className={`stack-order-btn ${stackOrderMode === "error" ? "active" : ""}`}
                                                onClick={() => setStackOrderMode("error")}
                                            >
                                                Errors Only
                                            </button>
                                        </div>
                                        {stackOrderMode !== "tree" && stackOrderMode !== "error" && (
                                            <Space align="center">
                                                <Switch size="small" checked={stackLeafOnly} onChange={setStackLeafOnly} />
                                                <span className="stack-leaf-toggle">Leaf only</span>
                                            </Space>
                                        )}
                                    </div>
                                    <div className="stack-legend">
                                        <span className="stack-legend-title">Legend</span>
                                        <span className="stack-legend-item"><span className="stack-legend-swatch normal" /> Node</span>
                                        <span className="stack-legend-item"><span className="stack-legend-swatch fail" /> Error</span>
                                        <span className="stack-legend-item"><span className="stack-legend-swatch path" /> Selected path</span>
                                        <span className="stack-legend-note">right edge = subtree group</span>
                                    </div>
                                    {selectedPathIds.length > 0 && (
                                        <div className="stack-pathbar">
                                            <span className="stack-path-label">Path</span>
                                            {selectedPathIds.map((id, index) => {
                                                const row = stackIndex?.rowMap.get(id);
                                                if (!row) return null;
                                                return (
                                                    <React.Fragment key={id}>
                                                        <button type="button" className="stack-path-chip" onClick={() => setSelectedId(id)}>
                                                            {row.node.name || row.node.kind || id}
                                                        </button>
                                                        {index < selectedPathIds.length - 1 && <span className="stack-path-sep">/</span>}
                                                    </React.Fragment>
                                                );
                                            })}
                                        </div>
                                    )}
                                </div>
                                <div className={`stack-table-head ${compactMode ? "compact" : ""}`}>
                                    <div className="stack-head-main">Node Stack</div>
                                    <div className="stack-head-metric">Time</div>
                                    <div className="stack-head-metric">Cost</div>
                                </div>
                                <div className="stack-list" ref={stackListRef}>
                                    {visibleStackRows.map((row) => {
                                        const hasChildren = row.childIds.length > 0;
                                        const isSelected = row.id === selectedId;
                                        const isFolded = isTreeStackMode && activeFoldIds.has(row.id);
                                        const inSelectedPath = selectedPathIds.includes(row.id);
                                        const tokenLine = row.node.kind === "LLM" ? formatTokenUsage(tokenUsageFromAttributes(row.node.attributes)) : "";
                                        const tone = getNodeTone(row.node.kind);
                                        const KindIcon = tone.icon;
                                        const subtreeBackground =
                                            row.subtreeColor && row.status !== "FAIL"
                                                ? `linear-gradient(to right, transparent 0%, ${hexToRgba(row.subtreeColor, compactMode ? 0.06 : 0.03)} 36%, ${hexToRgba(row.subtreeColor, compactMode ? 0.16 : 0.11)} 100%)`
                                                : undefined;
                                        const accentBackground =
                                            row.status === "FAIL"
                                                ? undefined
                                                : hasChildren
                                                  ? `linear-gradient(to right, ${hexToRgba(tone.accent, 0.18)} 0%, ${hexToRgba(tone.accent, 0.06)} 18%, rgba(255,255,255,0) 34%)`
                                                  : `linear-gradient(to right, ${hexToRgba(tone.accent, 0.14)} 0%, ${hexToRgba(tone.accent, 0.04)} 18%, rgba(255,255,255,0) 34%)`;
                                        return (
                                            <button
                                                key={row.id}
                                                type="button"
                                                data-stack-id={row.id}
                                                className={`stack-row ${compactMode ? "compact" : ""} ${isSelected ? "selected" : ""} ${inSelectedPath ? "path" : ""} ${row.status === "FAIL" ? "fail" : ""}`}
                                                style={{
                                                    borderRightColor: row.subtreeColor || undefined,
                                                    backgroundImage: [accentBackground, subtreeBackground].filter(Boolean).join(", "),
                                                }}
                                                onClick={() => setSelectedId(row.id)}
                                                onDoubleClick={() => {
                                                    if (!isTreeStackMode) return;
                                                    toggleFoldById(row.id, hasChildren, row.depth);
                                                }}
                                            >
                                                <div
                                                    className={`stack-row-main ${isTreeStackMode ? "" : "flat"}`}
                                                    style={{ paddingLeft: isTreeStackMode ? `${row.depth * 20 + 12}px` : "12px" }}
                                                >
                                                    {isTreeStackMode && (
                                                        <span
                                                            className={`stack-caret ${hasChildren ? "expandable" : "leaf"}`}
                                                            onClick={(event) => {
                                                                event.stopPropagation();
                                                                toggleFoldById(row.id, hasChildren, row.depth);
                                                            }}
                                                        >
                                                            {hasChildren ? (isFolded ? "+" : "-") : "·"}
                                                        </span>
                                                    )}
                                                    {row.subtreeColor && (
                                                        <span className="stack-subtree-dot" style={{ backgroundColor: row.subtreeColor }} />
                                                    )}
                                                    <div className="stack-main-copy">
                                                        <div className="stack-main-line">
                                                            <span
                                                                className={compactMode ? tone.compactClassName : tone.className}
                                                                style={{ "--tone-accent": tone.accent } as React.CSSProperties}
                                                            >
                                                                <span className="stack-kind-icon-shell">
                                                                    <KindIcon className="stack-kind-icon" />
                                                                </span>
                                                                {tone.label}
                                                            </span>
                                                            <span className="stack-name">{row.node.name || "(unnamed)"}</span>
                                                            <span className={`stack-status stack-status-${row.status.toLowerCase()}`}>{row.status}</span>
                                                        </div>
                                                        {!compactMode && tokenLine && <div className="stack-subline">{tokenLine}</div>}
                                                    </div>
                                                </div>
                                                <div className="stack-cell stack-cell-time">
                                                    <div className="stack-metric">
                                                        <div className="stack-metric-bar">
                                                            <div
                                                                className={`stack-metric-fill ${row.status === "FAIL" ? "fail" : "time"}`}
                                                                style={{
                                                                    width: stackStats.maxDuration > 0 ? `${Math.max(((row.node.duration || 0) / stackStats.maxDuration) * 100, row.node.duration > 0 ? 3 : 0)}%` : "0%",
                                                                }}
                                                            />
                                                        </div>
                                                        <div className="stack-metric-value">{formatStackDuration(row.node.duration || 0)}</div>
                                                    </div>
                                                </div>
                                                <div className="stack-cell stack-cell-cost">
                                                    <div className="stack-metric">
                                                        <div className="stack-metric-bar">
                                                            <div
                                                                className={`stack-metric-fill ${row.status === "FAIL" ? "fail" : "cost"}`}
                                                                style={{
                                                                    width: stackStats.maxCost > 0 ? `${Math.max(((row.node.cost || 0) / stackStats.maxCost) * 100, row.node.cost > 0 ? 3 : 0)}%` : "0%",
                                                                }}
                                                            />
                                                        </div>
                                                        <div className="stack-metric-value">{formatCost(row.node.cost || 0)}</div>
                                                    </div>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        )
                    ) : (
                        <div className="empty-state trace-home">
                            <Card className="trace-list-card" bordered>
                                <Space direction="vertical" size={16} style={{ width: "100%" }}>
                                    <div className="trace-list-header">
                                        <div>
                                            <Title level={4} className="details-title">
                                                Saved Traces
                                            </Title>
                                            <Text className="toggle-label">
                                                {traceListLoading
                                                    ? "Loading..."
                                                    : traceList.length
                                                      ? `${traceList.length} trace${traceList.length > 1 ? "s" : ""}`
                                                      : "No traces found in the current trace directory. Showing up to the most recent 100 traces."}
                                            </Text>
                                        </div>
                                        <Button size="small" onClick={() => void loadTraceList()} loading={traceListLoading}>
                                            Refresh
                                        </Button>
                                    </div>
                                    {traceList.length ? (
                                        <div className="trace-list">
                                            {traceList.map((item) => (
                                                <button
                                                    key={item.id}
                                                    type="button"
                                                    className="trace-list-item"
                                                    onClick={() => void loadTrace(item.id)}
                                                >
                                                    <div className="trace-list-main">
                                                        <div className="trace-list-name">{item.name || "(unnamed trace)"}</div>
                                                        <div className="trace-list-id">{item.id}</div>
                                                    </div>
                                                    <div className="trace-list-time">{formatTraceListDate(item.created_at)}</div>
                                                </button>
                                            ))}
                                        </div>
                                    ) : !traceListLoading ? (
                                        <div className="trace-list-empty">Run a tree and save its trace into the current trace directory.</div>
                                    ) : null}
                                </Space>
                            </Card>
                        </div>
                    )}
                </div>
                <div className="resizer" onMouseDown={resizer.onMouseDown} />
                <div className="right-panel">
                    {!selectedTrace ? (
                        <div className="empty-state">
                            {trace ? "Select a node to inspect details." : "Choose a trace from the list to inspect details."}
                        </div>
                    ) : (
                        <div className="details">
                            <Card className="details-header" bordered>
                                <Space align="start" className="details-title-row">
                                    <div>
                                        <Title level={4} className="details-title">
                                            {selectedTrace.name || "(unnamed)"}
                                        </Title>
                                        <Space size={8} align="center" className="details-sub">
                                            <Tag className="type-label">{selectedTrace.kind || "Unknown"}</Tag>
                                            <Tag color={statusColor(parseStatus(selectedTrace.result || null))}>
                                                {parseStatus(selectedTrace.result || null)}
                                            </Tag>
                                        </Space>
                                    </div>
                                    <div className="details-metrics">
                                        <Text className="details-cost">
                                            Cost: {formatCost(selectedTrace.cost || 0)}
                                        </Text>
                                        {(() => {
                                            if (selectedTrace.kind !== "LLM") return null;
                                            const usage = tokenUsageFromAttributes(selectedTrace.attributes);
                                            const tokenLine = formatTokenUsage(usage);
                                            return tokenLine ? (
                                                <Text className="details-tokens">{tokenLine}</Text>
                                            ) : null;
                                        })()}
                                    </div>
                                </Space>
                                <Descriptions
                                    className="details-grid"
                                    size="small"
                                    layout="horizontal"
                                    column={{ xs: 1, sm: 1, md: 1, lg: 1, xl: 1 }}
                                    items={detailsItems}
                                >
                                    {detailsItems.map((item) => (
                                        <Descriptions.Item
                                            key={item.key}
                                            label={item.label}
                                            span={1}
                                        >
                                            <span className="details-inline">{item.children}</span>
                                        </Descriptions.Item>
                                    ))}
                                </Descriptions>
                            </Card>

                            <Tabs
                                className="tabs"
                                activeKey={tab}
                                onChange={(key) => setTab(key as "result" | "logs" | "attributes")}
                                items={detailsTabs}
                            />

                        </div>
                    )}
                </div>
            </Content>
        </Layout>
    );
}

export default function App() {
    return (
        <ConfigProvider
            theme={{
                algorithm: theme.darkAlgorithm,
                token: {
                    colorPrimary: "#58a6ff",
                    colorInfo: "#58a6ff",
                    colorSuccess: "#3fb950",
                    colorError: "#f85149",
                    colorWarning: "#d29922",
                    colorBgBase: "#0d1117",
                    colorTextBase: "#c9d1d9",
                    borderRadius: 8,
                    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
                },
            }}
        >
            <TraceUI />
        </ConfigProvider>
    );
}
