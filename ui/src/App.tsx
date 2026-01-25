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
    AutoComplete,
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
    Tooltip,
    Typography,
} from "antd";
import "reactflow/dist/style.css";

const { Header, Content } = Layout;
const { Text, Title } = Typography;

const X_SPACING = 420;
const Y_SPACING = 190;
const MIN_LEFT_WIDTH = 360;
const MAX_LEFT_WIDTH = 1400;

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

type TraceStatus = "OK" | "FAIL" | "UNKNOWN";

type TraceNodeData = {
    node: TraceNodeJson;
    status: TraceStatus;
    durationMs: number;
    cost: number;
    subtreeColor: string | null;
    subtreeColored: boolean;
    isFolded: boolean;
    isSelected?: boolean;
};

type TokenUsage = {
    prompt?: number;
    completion?: number;
    total?: number;
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

function parseDate(value: string): number {
    const ts = Date.parse(value);
    return Number.isNaN(ts) ? 0 : ts;
}

function sortChildren(children: Record<string, TraceNodeJson>): TraceNodeJson[] {
    return Object.values(children).sort((a, b) => parseDate(a.start_at) - parseDate(b.start_at));
}

function extractSystemUser(messages: unknown): { system: string; user: string } {
    let system = "";
    let user = "";
    if (!Array.isArray(messages)) return { system, user };
    for (const msg of messages) {
        if (!msg || typeof msg !== "object") continue;
        const role = (msg as { role?: string }).role;
        const raw = (msg as { content?: unknown }).content;
        const content = typeof raw === "string" ? raw : raw != null ? JSON.stringify(raw) : "";
        if (role === "system" && !system) {
            system = content;
        }
        if (role === "user") {
            user = content;
        }
    }
    return { system, user };
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

function statusColor(status: TraceStatus): string {
    if (status === "OK") return "green";
    if (status === "FAIL") return "red";
    return "default";
}

function TraceCard(props: NodeProps<TraceNodeData>) {
    const { data, selected } = props;
    const isSelected = data.isSelected || selected;
    const status = data.status;
    let borderColor = "#1f7a4a";
    if (status === "FAIL") borderColor = "#d11f1f";
    if (status === "UNKNOWN") borderColor = "#7b7b7b";
    if (isSelected) borderColor = "#000000";
    const borderWidth = isSelected ? 6 : status === "FAIL" ? 5 : 4;
    const background =
        data.subtreeColored && data.subtreeColor && data.subtreeColor !== null
            ? hexToRgba(data.subtreeColor, 0.14)
            : "#ffffff";
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
                {foldedPrefix}
                {data.node.name || "(unnamed)"}
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
    const [loadError, setLoadError] = useState<string | null>(null);
    const [selectedId, setSelectedId] = useState<string | null>(null);
    const [tab, setTab] = useState<"result" | "logs" | "attributes">("result");
    const [playModel, setPlayModel] = useState("");
    const [playMessages, setPlayMessages] = useState("[]");
    const [playEditorMode, setPlayEditorMode] = useState<"split" | "json">("split");
    const [playSystem, setPlaySystem] = useState("");
    const [playUser, setPlayUser] = useState("");
    const [playStream, setPlayStream] = useState(false);
    const [playOutput, setPlayOutput] = useState("");
    const [playRunning, setPlayRunning] = useState(false);
    const [playError, setPlayError] = useState<string | null>(null);
    const initialLeftWidth = useMemo(() => {
        if (typeof window === "undefined") return 640;
        return Math.min(MAX_LEFT_WIDTH, Math.max(MIN_LEFT_WIDTH, Math.floor(window.innerWidth * 0.65)));
    }, []);
    const resizer = useResizableSidebar(initialLeftWidth);
    const [searchTerm, setSearchTerm] = useState("");
    const [searchOpen, setSearchOpen] = useState(false);
    const ignoreSearchChange = useRef(false);
    const [flowInstance, setFlowInstance] = useState<any>(null);
    const fitTimer = useRef<number | null>(null);
    const [subtreeColorOn, setSubtreeColorOn] = useState(true);
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
        const tokens = query
            .trim()
            .toLowerCase()
            .split(/\s+/)
            .filter(Boolean);
        if (!tokens.length) return null;
        for (const token of tokens) {
            const idx = hay.indexOf(token);
            if (idx !== -1) return [idx];
        }
        return null;
    }, []);

    useEffect(() => {
        const pathId = window.location.pathname.replace(/^\/+/, "");
        if (pathId) {
            void loadTrace(pathId);
        }
        // no selector to refresh
    }, []);

    const loadTrace = useCallback(async (id: string) => {
        setLoadError(null);
        setTrace(null);
        setSelectedId(null);
        try {
            const resp = await fetch(`/trace/${encodeURIComponent(id)}`);
            if (!resp.ok) {
                throw new Error(`Failed to load trace: ${resp.status}`);
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

    const { nodes, edges, nodeMap, searchIndex } = useMemo(() => {
        if (!trace) {
            return {
                nodes: [] as Node<TraceNodeData>[],
                edges: [] as Edge[],
                nodeMap: new Map<string, FlattenedNode>(),
                searchIndex: [] as { id: string; text: string }[],
            };
        }
        const { flat, pos } = buildLayout(trace, maxDepth, activeFoldIds);
        const nodeMapLocal = new Map<string, FlattenedNode>();
        flat.forEach((item) => nodeMapLocal.set(item.id, item));
        const childrenByParent = new Map<string | null, string[]>();
        flat.forEach((item) => {
            const list = childrenByParent.get(item.parentId) || [];
            list.push(item.id);
            childrenByParent.set(item.parentId, list);
        });
        const subtreeColorMap = new Map<string, string | null>();
        let colorIdx = 0;

        const depthMap = new Map<string, number>();
        flat.forEach((item) => depthMap.set(item.id, item.depth));

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

        subtreeRoots.forEach((rootId) => {
            const color = SUBTREE_COLORS[colorIdx % SUBTREE_COLORS.length];
            colorIdx += 1;
            assignSubtree(rootId, color);
        });

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
                    subtreeColored: subtreeColorOn,
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
        const searchIndexLocal = flat.map((item) => {
            const blob = JSON.stringify({
                name: item.node.name,
                kind: item.node.kind,
                result: item.node.result,
                logs: item.node.logs,
                attributes: item.node.attributes,
            });
            return { id: item.id, text: blob };
        });
        return { nodes: nodesLocal, edges: edgesLocal, nodeMap: nodeMapLocal, searchIndex: searchIndexLocal };
    }, [trace, subtreeColorOn, maxDepth, autoFoldOn, activeFoldIds, selectedId]);

    const selectedNode = selectedId ? nodeMap.get(selectedId) : null;
    const selectedTrace = selectedNode?.node || null;
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

    useEffect(() => {
        if (!selectedTrace) return;
        const modelStr = selectedTrace.attributes?.model ? selectedTrace.attributes.model : "";
        const messagesStr = selectedTrace.attributes?.messages ? selectedTrace.attributes.messages : "[]";
        try {
            const parsedModel = JSON.parse(modelStr);
            if (typeof parsedModel === "string") setPlayModel(parsedModel);
        } catch {
            if (modelStr) setPlayModel(modelStr);
        }
        try {
            const parsedMessages = JSON.parse(messagesStr);
            setPlayMessages(JSON.stringify(parsedMessages, null, 2));
            const extracted = extractSystemUser(parsedMessages);
            setPlaySystem(extracted.system);
            setPlayUser(extracted.user);
        } catch {
            setPlayMessages(messagesStr || "[]");
            setPlaySystem("");
            setPlayUser("");
        }
        const streamAttr = selectedTrace.attributes?.stream;
        if (typeof streamAttr === "boolean") {
            setPlayStream(streamAttr);
        } else if (typeof streamAttr === "number") {
            setPlayStream(streamAttr !== 0);
        } else if (typeof streamAttr === "string") {
            const s = streamAttr.trim().toLowerCase();
            if (s === "true" || s === "1" || s === "yes") {
                setPlayStream(true);
            } else if (s === "false" || s === "0" || s === "no") {
                setPlayStream(false);
            } else {
                setPlayStream(false);
            }
        } else {
            setPlayStream(false);
        }
    }, [selectedTrace]);

    const onRunPlayground = useCallback(async () => {
        setPlayRunning(true);
        setPlayOutput("");
        setPlayError(null);
        try {
            let messages: any[] = [];
            if (playEditorMode === "split") {
                const sys = playSystem.trim();
                const usr = playUser.trim();
                messages = [];
                if (sys) messages.push({ role: "system", content: sys });
                if (usr) messages.push({ role: "user", content: usr });
            } else {
                messages = JSON.parse(playMessages);
                if (!Array.isArray(messages)) {
                    throw new Error("Messages JSON must be an array");
                }
            }
            const payload = {
                model: playModel,
                messages,
                stream: playStream,
            };
            const resp = await fetch("/llm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                let detail = `LLM request failed: ${resp.status}`;
                try {
                    const errBody = await resp.json();
                    if (errBody?.error) detail = String(errBody.error);
                } catch {
                    try {
                        const errText = await resp.text();
                        if (errText) detail = errText;
                    } catch {
                        // ignore
                    }
                }
                throw new Error(detail);
            }
            if (!playStream) {
                const data = await resp.json();
                setPlayOutput(data.output || "");
                return;
            }
            const reader = resp.body?.getReader();
            if (!reader) {
                throw new Error("Streaming not supported");
            }
            const decoder = new TextDecoder();
            let done = false;
            while (!done) {
                const chunk = await reader.read();
                done = chunk.done;
                if (chunk.value) {
                    setPlayOutput((prev) => prev + decoder.decode(chunk.value, { stream: true }));
                }
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setPlayError(message);
            setPlayOutput(`Error: ${message}`);
        } finally {
            setPlayRunning(false);
        }
    }, [playEditorMode, playModel, playMessages, playStream, playSystem, playUser]);

    const total = trace ? totalCost(trace) : 0;
    const searchResults = useMemo(() => {
        if (!searchTerm.trim()) return [];
        const results: { id: string; name: string; hits: number[]; nameHits: number[] | null }[] = [];
        for (const item of searchIndex) {
            const node = nodeMap.get(item.id)?.node;
            if (!node) continue;
            const hits = fuzzyMatch(item.text, searchTerm.trim());
            if (hits) {
                const nameHits = node.name ? fuzzyMatch(node.name, searchTerm.trim()) : null;
                results.push({ id: item.id, name: node.name || "(unnamed)", hits, nameHits });
            }
        }
        return results
            .sort((a, b) => {
                const aName = a.nameHits ? 1 : 0;
                const bName = b.nameHits ? 1 : 0;
                if (aName !== bName) return bName - aName;
                return a.hits.length - b.hits.length;
            })
            .slice(0, 12);
    }, [searchIndex, searchTerm, nodeMap, fuzzyMatch]);

    const highlightText = useCallback((text: string, query: string) => {
        const tokens = query
            .trim()
            .toLowerCase()
            .split(/\s+/)
            .filter(Boolean);
        if (!tokens.length) return text;
        const parts = text.split(new RegExp(`(${tokens.join("|")})`, "gi"));
        return parts.map((part, idx) => {
            if (tokens.includes(part.toLowerCase())) {
                return (
                    <mark key={`${idx}-${part}`} className="search-hit">
                        {part}
                    </mark>
                );
            }
            return <span key={`${idx}-${part}`}>{part}</span>;
        });
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

    const handleTogglePlayEditor = useCallback(() => {
        if (playEditorMode === "split") {
            const messages: any[] = [];
            const sys = playSystem.trim();
            const usr = playUser.trim();
            if (sys) messages.push({ role: "system", content: sys });
            if (usr) messages.push({ role: "user", content: usr });
            setPlayMessages(JSON.stringify(messages, null, 2));
            setPlayEditorMode("json");
            return;
        }
        try {
            const parsed = JSON.parse(playMessages);
            const extracted = extractSystemUser(parsed);
            setPlaySystem(extracted.system);
            setPlayUser(extracted.user);
            setPlayEditorMode("split");
        } catch {
            setPlayError("Invalid JSON: cannot auto-format");
        }
    }, [playEditorMode, playMessages, playSystem, playUser]);

    const searchOptions = useMemo(() => {
        return searchResults.map((item) => {
            const content = searchIndex.find((entry) => entry.id === item.id)?.text || "";
            const snippet = buildSnippet(content, searchTerm, item.hits);
            return {
                value: item.name || "(unnamed)",
                id: item.id,
                label: (
                    <div className="search-item">
                        <div className="search-name">{highlightText(item.name, searchTerm)}</div>
                        <div className="search-snippet">
                            {snippet ? highlightText(snippet, searchTerm) : "-"}
                        </div>
                    </div>
                ),
            };
        });
    }, [searchResults, searchIndex, buildSnippet, highlightText, searchTerm]);

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
                                    width: "35%",
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
                    <Title level={4} className="brand">
                        tinytasktree Trace UI
                    </Title>
                </Space>
                <div className="search-box">
                    <AutoComplete
                        value={searchTerm}
                        options={searchOptions}
                        open={searchOpen && searchResults.length > 0}
                        onChange={(value) => {
                            if (ignoreSearchChange.current) {
                                ignoreSearchChange.current = false;
                                return;
                            }
                            setSearchTerm(value);
                            setSearchOpen(true);
                        }}
                        onSelect={(value, option) => {
                            const opt = option as { id?: string };
                            const selectedId = opt?.id ?? value;
                            ignoreSearchChange.current = true;
                            setSearchTerm(value);
                            setSelectedId(selectedId);
                            requestAnimationFrame(() => focusNodeById(selectedId));
                            setSearchOpen(false);
                        }}
                        className="search-input"
                    >
                        <Input
                            size="large"
                            placeholder="Search nodes..."
                            onFocus={() => setSearchOpen(true)}
                            onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
                        />
                    </AutoComplete>
                </div>
                <Space size={16} align="center" className="top-controls">
                    <Space direction="vertical" size={4} className="toggle-stack">
                        <Space align="center">
                            <Switch checked={autoFoldOn} onChange={setAutoFoldOn} size="small" />
                            <Text className="toggle-label">Auto fold</Text>
                        </Space>
                        <Text className="top-hint">Double-click a node to fold/unfold its subtree</Text>
                    </Space>
                    <Tooltip title="Color subtrees by top-level Tree nodes">
                        <Space align="center">
                            <Switch checked={subtreeColorOn} onChange={setSubtreeColorOn} size="small" />
                            <Text className="toggle-label">Subtree color</Text>
                        </Space>
                    </Tooltip>
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
                                    if (!hasChildren) return;
                                    const depthLimited = maxDepth !== null && target.depth >= maxDepth;
                                    if (depthLimited) return;
                                    const isSuggested = suggestFoldIds.has(node.id);
                                    const isManualFolded = manualFoldIds.has(node.id);
                                    const isExpandedOverride = manualExpandIds.has(node.id);
                                    const isFolded =
                                        isManualFolded || (autoFoldOn && isSuggested && !isExpandedOverride);

                                    if (isFolded) {
                                        setManualFoldIds((prev) => {
                                            const next = new Set(prev);
                                            next.delete(node.id);
                                            return next;
                                        });
                                        if (isSuggested) {
                                            setManualExpandIds((prev) => new Set(prev).add(node.id));
                                        }
                                    } else {
                                        setManualExpandIds((prev) => {
                                            const next = new Set(prev);
                                            next.delete(node.id);
                                            return next;
                                        });
                                        setManualFoldIds((prev) => new Set(prev).add(node.id));
                                    }
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
                                    maskColor="rgba(200, 200, 200, 0.8)"
                                    style={{ backgroundColor: "#ffffff", border: "2px solid #000000" }}
                                    nodeColor={(node) => {
                                        if ((node.data as TraceNodeData)?.isSelected || node.selected) return "#000000";
                                        const data = node.data as TraceNodeData;
                                        if (data.status === "FAIL") return "#d11f1f";
                                        if (subtreeColorOn && data.subtreeColor) {
                                            return toPastel(data.subtreeColor, 0.35);
                                        }
                                        return "#888888";
                                    }}
                                />
                                <Controls showInteractive={false} />
                            </ReactFlow>
                        </ReactFlowProvider>
                    ) : (
                        <div className="empty-state">Load a trace to see the execution flow.</div>
                    )}
                </div>
                <div className="resizer" onMouseDown={resizer.onMouseDown} />
                <div className="right-panel">
                    {!selectedTrace ? (
                        <div className="empty-state">Select a node to inspect details.</div>
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

                            {selectedTrace.kind === "LLM" && (
                                <Card
                                    className="playground"
                                    title={<Text strong>LLM Playground</Text>}
                                    extra={
                                        <Button size="small" onClick={handleTogglePlayEditor}>
                                            {playEditorMode === "split" ? "JSON" : "Text"}
                                        </Button>
                                    }
                                >
                                    <Space direction="vertical" size={12} className="playground-fields">
                                        <div>
                                            <Text className="field-label">Model</Text>
                                            <Input value={playModel} onChange={(e) => setPlayModel(e.target.value)} />
                                        </div>
                                        {playEditorMode === "split" ? (
                                            <>
                                                <div>
                                                    <Text className="field-label">System</Text>
                                                    <Input.TextArea
                                                        className="playground-textarea"
                                                        rows={8}
                                                        value={playSystem}
                                                        onChange={(e) => setPlaySystem(e.target.value)}
                                                    />
                                                </div>
                                                <div>
                                                    <Text className="field-label">User</Text>
                                                    <Input.TextArea
                                                        className="playground-textarea"
                                                        rows={10}
                                                        value={playUser}
                                                        onChange={(e) => setPlayUser(e.target.value)}
                                                    />
                                                </div>
                                            </>
                                        ) : (
                                            <div>
                                                <Text className="field-label">Messages (JSON)</Text>
                                                <Input.TextArea
                                                    className="playground-textarea"
                                                    rows={12}
                                                    value={playMessages}
                                                    onChange={(e) => setPlayMessages(e.target.value)}
                                                />
                                            </div>
                                        )}
                                        <Space align="center">
                                            <Switch checked={playStream} onChange={setPlayStream} size="small" />
                                            <Text className="toggle-label">Stream</Text>
                                        </Space>
                                        <Space align="center">
                                            <Button type="primary" onClick={onRunPlayground} loading={playRunning}>
                                                {playRunning ? "Running..." : "Run"}
                                            </Button>
                                            {playError && <Text className="error-inline">{playError}</Text>}
                                        </Space>
                                    </Space>
                                    <div className="playground-output">
                                        {playError && (
                                            <Alert type="error" message={playError} showIcon className="playground-error" />
                                        )}
                                        <div className="panel-actions">
                                            <Button size="small" onClick={() => copyText(playOutput)}>
                                                Copy
                                            </Button>
                                        </div>
                                        <pre>{playOutput || "(no output yet)"}</pre>
                                    </div>
                                </Card>
                            )}
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
                token: {
                    colorPrimary: "#0b5d47",
                    colorInfo: "#0b5d47",
                    colorSuccess: "#0f7a3c",
                    colorError: "#b40000",
                    borderRadius: 8,
                    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace",
                },
            }}
        >
            <TraceUI />
        </ConfigProvider>
    );
}
