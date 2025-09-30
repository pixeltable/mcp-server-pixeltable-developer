import { useCallback, useEffect, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  BackgroundVariant,
  MiniMap,
} from 'reactflow';
import 'reactflow/dist/style.css';

interface PipelineViewProps {
  tables: string[];
  onSelectTable: (table: string) => void;
}

export const PipelineView = ({ tables, onSelectTable }: PipelineViewProps) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    // Convert tables to nodes
    const newNodes: Node[] = tables.map((table, index) => ({
      id: table,
      type: 'default',
      data: { label: table },
      position: { x: 100 + (index % 3) * 250, y: 100 + Math.floor(index / 3) * 150 },
      style: {
        background: '#1f2937',
        color: '#fff',
        border: '1px solid #4b5563',
        borderRadius: '8px',
        padding: '10px',
        cursor: 'pointer',
      },
    }));

    setNodes(newNodes);
  }, [tables, setNodes]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      onSelectTable(node.id);
    },
    [onSelectTable]
  );

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        fitView
      >
        <Controls />
        <MiniMap
          nodeColor="#1f2937"
          maskColor="rgba(0, 0, 0, 0.5)"
          style={{ background: '#111827' }}
        />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} color="#4b5563" />
      </ReactFlow>
    </div>
  );
};
