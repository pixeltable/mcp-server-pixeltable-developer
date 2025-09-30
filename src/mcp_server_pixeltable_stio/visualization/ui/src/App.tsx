import { useEffect, useState } from 'react';
import { PipelineView } from './PipelineView';
import { useWebSocket } from './useWebSocket';

interface TableInfo {
  name: string;
  schema: Record<string, { type: string }>;
  row_count: number;
}

function App() {
  const [tables, setTables] = useState<string[]>([]);
  const [selectedTable, setSelectedTable] = useState<TableInfo | null>(null);
  const { isConnected, lastMessage, sendMessage } = useWebSocket('/ws/pipeline');

  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'tables') {
        setTables(lastMessage.data || []);
      } else if (lastMessage.type === 'table_info') {
        setSelectedTable(lastMessage.data);
      }
    }
  }, [lastMessage]);

  const handleSelectTable = (table: string) => {
    sendMessage({ type: 'get_table', table_path: table });
  };

  return (
    <div className="w-full h-screen bg-gray-900 text-white flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Pixeltable Visualization</h1>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span className="text-sm text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Pipeline View */}
        <div className="flex-1">
          {tables.length > 0 ? (
            <PipelineView tables={tables} onSelectTable={handleSelectTable} />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">
                {isConnected ? 'No tables found' : 'Connecting...'}
              </p>
            </div>
          )}
        </div>

        {/* Table Details Sidebar */}
        {selectedTable && (
          <aside className="w-96 bg-gray-800 border-l border-gray-700 p-6 overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">{selectedTable.name}</h2>

            <div className="mb-6">
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Stats</h3>
              <p className="text-sm">
                <span className="text-gray-400">Row Count:</span>{' '}
                {selectedTable.row_count}
              </p>
            </div>

            <div>
              <h3 className="text-sm font-semibold text-gray-400 mb-2">Schema</h3>
              <div className="space-y-2">
                {Object.entries(selectedTable.schema).map(([colName, colInfo]) => (
                  <div
                    key={colName}
                    className="bg-gray-700 rounded px-3 py-2 text-sm"
                  >
                    <div className="font-mono text-blue-400">{colName}</div>
                    <div className="text-gray-400 text-xs mt-1">
                      {colInfo.type}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
