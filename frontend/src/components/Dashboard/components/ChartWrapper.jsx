import React, { useRef, useEffect } from 'react';
import { Pie, Bar } from 'react-chartjs-2';

// Wrapper component để handle chart cleanup
export const ChartWrapper = ({ type, data, options, style }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    const chartInstance = chartRef.current;
    return () => {
      if (chartInstance) {
        try {
          chartInstance.destroy();
        } catch (e) {
          console.warn('Error destroying chart:', e);
        }
      }
    };
  }, []);

  const ChartComponent = type === 'pie' ? Pie : Bar;

  return (
    <div style={style}>
      <ChartComponent
        ref={chartRef}
        data={data}
        options={{
          ...options,
          // Force chart to be responsive and destroy on unmount
          responsive: true,
          maintainAspectRatio: false,
        }}
      />
    </div>
  );
};

export default ChartWrapper;
