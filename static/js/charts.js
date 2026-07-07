/**
 * charts.js — Chart.js Integrations
 * Manages Spread Over Time (ABM vs Real data) and SLP Loss charts.
 */

const ChartManager = (() => {
    let spreadChart = null;
    let lossChart = null;
    let ebolaChart = null;

    const COLORS = {
        S: '#43A047',
        E: '#FFA726',
        I: '#E53935',
        R: '#1E88E5',
        D: '#9E9E9E',
        REAL: '#FF8A65'
    };

    function init() {
        initSpreadChart();
        initLossChart();
        initEbolaChart();
    }

    function initSpreadChart() {
        const ctx = document.getElementById('spreadChart').getContext('2d');
        spreadChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Susceptible', data: [], borderColor: COLORS.S, backgroundColor: COLORS.S + '20', fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Exposed', data: [], borderColor: COLORS.E, backgroundColor: 'transparent', tension: 0.4, pointRadius: 0 },
                    { label: 'Infected', data: [], borderColor: COLORS.I, backgroundColor: COLORS.I + '20', fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Recovered', data: [], borderColor: COLORS.R, backgroundColor: 'transparent', tension: 0.4, pointRadius: 0 },
                    { label: 'Dead', data: [], borderColor: COLORS.D, backgroundColor: 'transparent', tension: 0.4, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: true }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true }
                }
            }
        });
    }

    function initLossChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Loss', data: [], borderColor: '#8E24AA', backgroundColor: '#8E24AA20', fill: true, tension: 0.4, pointRadius: 0 },
                    { label: 'Accuracy', data: [], borderColor: '#43A047', backgroundColor: 'transparent', tension: 0.4, pointRadius: 0, yAxisID: 'y1' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top', labels: { boxWidth: 12 } } },
                scales: {
                    x: { display: false },
                    y: { type: 'linear', display: true, position: 'left' },
                    y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, min: 0, max: 1 }
                }
            }
        });
    }

    function initEbolaChart() {
        const ctx = document.getElementById('ebolaChart').getContext('2d');
        ebolaChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Cumulative Cases', data: [], borderColor: COLORS.REAL, backgroundColor: COLORS.REAL + '20', fill: true, tension: 0.1, pointRadius: 0 },
                    { label: 'Cumulative Deaths', data: [], borderColor: COLORS.D, backgroundColor: 'transparent', tension: 0.1, pointRadius: 0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'top' } },
                scales: {
                    x: { ticks: { maxTicksLimit: 10 } },
                    y: { beginAtZero: true }
                }
            }
        });
    }

    function updateSpreadChart(history) {
        if (!spreadChart || !history || history.length === 0) return;

        const labels = history.map(h => `Day ${h.step}`);
        const s = history.map(h => h.S);
        const e = history.map(h => h.E);
        const i = history.map(h => h.I);
        const r = history.map(h => h.R);
        const d = history.map(h => h.D);

        spreadChart.data.labels = labels;
        spreadChart.data.datasets[0].data = s;
        spreadChart.data.datasets[1].data = e;
        spreadChart.data.datasets[2].data = i;
        spreadChart.data.datasets[3].data = r;
        spreadChart.data.datasets[4].data = d;
        spreadChart.update();
    }

    function updateSlpChart(info) {
        if (!lossChart || !info || !info.loss_history) return;

        const epochs = Array.from({length: info.loss_history.length}, (_, i) => i + 1);
        lossChart.data.labels = epochs;
        lossChart.data.datasets[0].data = info.loss_history;
        lossChart.data.datasets[1].data = info.accuracy_history;
        lossChart.update();
    }

    function updateEbolaChart(data) {
        if (!ebolaChart || !data || !data.timeseries) return;

        ebolaChart.data.labels = data.timeseries.dates;
        ebolaChart.data.datasets[0].data = data.timeseries.cases;
        ebolaChart.data.datasets[1].data = data.timeseries.deaths;
        ebolaChart.update();
    }

    function resetSpreadChart() {
        if (!spreadChart) return;
        spreadChart.data.labels = [];
        spreadChart.data.datasets.forEach(ds => ds.data = []);
        spreadChart.update();
    }

    return { init, updateSpreadChart, updateSlpChart, updateEbolaChart, resetSpreadChart };
})();
