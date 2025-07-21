import React, { useRef, useEffect } from 'react';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart, ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

// Đăng ký các thành phần cần thiết cho Chart.js
Chart.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

// Wrapper component để handle chart cleanup
const ChartWrapper = ({ type, data, options, style }) => {
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
