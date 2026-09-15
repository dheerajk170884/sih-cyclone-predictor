// Wind Speed Chart
    const windChart = new Chart(document.getElementById('windChart'), {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Wind speed',
          data: [],
          borderColor: '#3B82F6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#F59E0B',
          pointBorderColor: '#F8FAFC',
          pointBorderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.2,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: true, labels: { color: '#CBD5E1' } },
          tooltip: {
            callbacks: {
              label: (context) => ` ${Number(context.raw).toFixed(1)} kt`
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: { display: true, text: 'Prediction branch', color: '#94A3B8' },
            ticks: { color: '#94A3B8' },
            grid: { color: 'rgba(148, 163, 184, 0.16)' }
          },
          y: {
            display: true,
            title: { display: true, text: 'Wind speed (kt)', color: '#94A3B8' },
            ticks: { color: '#94A3B8' },
            grid: { color: 'rgba(148, 163, 184, 0.16)' },
            beginAtZero: true
          }
        }
      }
    });

    const predictionForm = document.getElementById('predictionForm');
    const predictionMessage = document.getElementById('predictionMessage');
    const confidenceValue = document.getElementById('confidenceValue');
    const categoryValue = document.getElementById('categoryValue');
    const trackValue = document.getElementById('trackValue');
    const impactValue = document.getElementById('impactValue');
    const riskValue = document.getElementById('riskValue');
    const intensityLabels = [
      'Tropical storm',
      'Tropical typhoon',
      'Violent tropical typhoon'
    ];

    const cycloneMap = L.map('cycloneMap', {
      zoomControl: true,
      worldCopyJump: true
    }).setView([15, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 18
    }).addTo(cycloneMap);
    let trackLayer = null;
    let impactLayer = null;

    function updateMap(track, impactRadiusKm) {
      if (!track?.length) return;
      if (trackLayer) trackLayer.remove();
      if (impactLayer) impactLayer.remove();

      const points = track.map((point) => [
        Number(point.latitude),
        Number(point.longitude_normalized ?? point.longitude)
      ]);
      trackLayer = L.layerGroup().addTo(cycloneMap);
      L.polyline(points, {
        color: '#ef4444',
        weight: 4,
        opacity: 0.9
      }).addTo(trackLayer);

      track.forEach((point, index) => {
        const position = points[index];
        const isCurrent = index === track.length - 1;
        L.circleMarker(position, {
          radius: isCurrent ? 8 : 5,
          color: isCurrent ? '#f59e0b' : '#2563eb',
          fillColor: isCurrent ? '#f59e0b' : '#60a5fa',
          fillOpacity: 1,
          weight: 2
        }).bindPopup(
          `<strong>${isCurrent ? 'Current position' : `Track point ${index + 1}`}</strong><br>` +
          `Time: ${point.time}<br>Lat: ${Number(point.latitude).toFixed(2)}<br>` +
          `Lon: ${Number(point.longitude_normalized ?? point.longitude).toFixed(2)}`
        ).addTo(trackLayer);
      });

      const currentPoint = points[points.length - 1];
      impactLayer = L.layerGroup().addTo(cycloneMap);
      const impactCircle = L.circle(currentPoint, {
        radius: Number(impactRadiusKm) * 1000,
        color: '#f59e0b',
        weight: 2,
        fillColor: '#f59e0b',
        fillOpacity: 0.16
      }).bindPopup(
        `<strong>Estimated impact area</strong><br>` +
        `Radius: ${Number(impactRadiusKm).toFixed(1)} km`
      ).addTo(impactLayer);

      const bounds = L.latLngBounds(points);
      bounds.extend(impactCircle.getBounds());
      cycloneMap.fitBounds(bounds, { padding: [24, 24], maxZoom: 6 });
      window.setTimeout(() => cycloneMap.invalidateSize(), 0);
    }

    async function runPrediction(url, options = {}) {
      predictionMessage.textContent = 'Running prediction...';

      try {
        const response = await fetch(url, options);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Prediction failed.');

        if (result.is_cyclone === false) {
          categoryValue.textContent = 'No Cyclone Detected';
          confidenceValue.textContent = '';
          trackValue.textContent = 'Awaiting cyclone confirmation';
          impactValue.textContent = 'Awaiting cyclone confirmation';
          riskValue.textContent = 'No cyclone';
          windChart.data.datasets[0].data = [];
          windChart.data.labels = [];
          windChart.update();
          predictionMessage.textContent = result.message;
          return;
        }

        const windSpeed = result.final_vmax_kt[0];
        const classIndex = Number(result.final_class_index[0]);
        confidenceValue.textContent = `${Number(windSpeed).toFixed(1)} kt`;
        categoryValue.textContent = intensityLabels[classIndex] || 'Unknown intensity';
        riskValue.textContent = `Class ${classIndex}`;
        updateTrackValue(result);
        impactValue.textContent = `${Number(result.impact_area_km2).toLocaleString()} km² (${Number(result.impact_radius_km).toFixed(1)} km radius)`;
        updateMap(result.track, result.impact_radius_km);
        windChart.data.datasets[0].data = [
          result.branch_vmax_kt.IR,
          result.branch_vmax_kt.WV,
          result.branch_vmax_kt.VIS,
          windSpeed
        ];
        windChart.data.labels = ['IR', 'WV', 'VIS', 'Fused'];
        windChart.update();

        predictionMessage.textContent = result.message || 'Prediction complete.';
      } catch (error) {
        predictionMessage.textContent = error.message;
      }
    }

    function updateTrackValue(result) {
      const current = result.track?.[result.track.length - 1];
      if (!current) return;
      const longitude = Number(current.longitude_normalized ?? current.longitude).toFixed(1);
      trackValue.textContent = `${longitude}, ${Number(current.latitude).toFixed(1)}`;
    }

    predictionForm.addEventListener('submit', (event) => {
      event.preventDefault();
      const formData = new FormData(predictionForm);
      if (!formData.get('visible')?.name || !formData.get('infrared')?.name || !formData.get('water_vapour')?.name || !formData.get('track')?.name) {
        predictionMessage.textContent = 'Select all three images and the cyclone track CSV.';
        return;
      }
      runPrediction('/api/run-predict', { method: 'POST', body: formData });
    });
