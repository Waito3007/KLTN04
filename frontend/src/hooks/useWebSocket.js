// e:\Project\KLTN04-1\frontend\src\hooks\useWebSocket.js
import { useEffect, useRef } from 'react';

const useWebSocket = (url, onMessage) => {
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };

    return () => {
      ws.close();
    };
  }, [url, onMessage]);

  return wsRef.current;
};

export default useWebSocket;
