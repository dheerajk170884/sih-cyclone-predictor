// Wind Speed Chart
    const windChart = new Chart(document.getElementById('windChart'), {
      type: 'line',
      data: {
        labels: ['00:00', '04:00', '08:00', '12:00', '16:00'],
        datasets: [{
          data: [90, 110, 125, 135, 145],
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }
    });

    // Barometric Pressure Chart
    const pressureChart = new Chart(document.getElementById('pressureChart'), {
      type: 'line',
      data: {
        labels: ['00:00', '04:00', '08:00', '12:00', '16:00'],
        datasets: [{
          data: [980, 972, 965, 958, 950],
          borderColor: '#EF4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.4
        }]
      },
      options: { plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: false } } }
    });

    const predictionForm = document.getElementById('predictionForm');
    const predictionMessage = document.getElementById('predictionMessage');
    const confidenceValue = document.getElementById('confidenceValue');
    const categoryValue = document.getElementById('categoryValue');
    const cycloneValue = document.getElementById('cycloneValue');

    predictionForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      predictionMessage.textContent = 'Running prediction...';

      try {
        const response = await fetch('/api/demo-predict', { method: 'POST' });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Prediction failed.');

        const windSpeed = result.final_vmax_kt[0];
        confidenceValue.textContent = `${Number(windSpeed).toFixed(1)} kt`;
        categoryValue.textContent = result.final_class[0];
        cycloneValue.textContent = `${result.estimated_pressure_drop_hpa.toFixed(1)} hPa`;
        windChart.data.datasets[0].data = [
          result.branch_vmax_kt.IR,
          result.branch_vmax_kt.WV,
          result.branch_vmax_kt.VIS,
          windSpeed
        ];
        windChart.data.labels = ['IR', 'WV', 'VIS', 'Fused'];
        windChart.update();

        const pressureDrop = result.estimated_pressure_drop_hpa;
        pressureChart.data.datasets[0].data = [0, pressureDrop * 0.35, pressureDrop * 0.65, pressureDrop];
        pressureChart.data.labels = ['IR', 'WV', 'VIS', 'Fused'];
        pressureChart.update();
        predictionMessage.textContent = 'Prediction complete.';
      } catch (error) {
        predictionMessage.textContent = error.message;
      }
    });