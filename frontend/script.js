// Wind Speed Chart
    new Chart(document.getElementById('windChart'), {
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
    new Chart(document.getElementById('pressureChart'), {
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