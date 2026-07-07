/**
 * sim.js — Agent Canvas Renderer
 * Draws agent dots with glow effects on HTML5 Canvas.
 * Color coded by SEIRD state.
 */

const SimRenderer = (() => {
    let canvas, ctx;
    let agents = [];
    let canvasSize = 500;

    const COLORS = {
        S: '#43A047',  // Green - Susceptible/Healthy
        E: '#FFA726',  // Orange - Exposed
        I: '#E53935',  // Red - Infectious
        R: '#1E88E5',  // Blue - Recovered
        D: '#9E9E9E',  // Gray - Dead
    };

    const GLOW = {
        I: 'rgba(229, 57, 53, 0.35)',
        E: 'rgba(255, 167, 38, 0.25)',
    };

    function init() {
        canvas = document.getElementById('agentCanvas');
        if (!canvas) return;
        ctx = canvas.getContext('2d');
        resize();
        window.addEventListener('resize', resize);
    }

    function resize() {
        const wrapper = canvas.parentElement;
        const size = Math.min(wrapper.clientWidth, wrapper.clientHeight);
        canvasSize = size;
        canvas.width = size * window.devicePixelRatio;
        canvas.height = size * window.devicePixelRatio;
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
        draw();
    }

    function setAgents(agentData) {
        agents = agentData || [];
        draw();
    }

    function draw() {
        if (!ctx) return;
        const s = canvasSize;

        // Clear with dark background
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, s, s);

        // Scale: model space is 100x100
        const scale = s / 100;

        // Draw dead agents first (back layer)
        agents.filter(a => a.state === 'D').forEach(a => drawAgent(a, scale, s));
        // Then susceptible
        agents.filter(a => a.state === 'S').forEach(a => drawAgent(a, scale, s));
        // Then recovered
        agents.filter(a => a.state === 'R').forEach(a => drawAgent(a, scale, s));
        // Then exposed
        agents.filter(a => a.state === 'E').forEach(a => drawAgent(a, scale, s));
        // Infectious on top
        agents.filter(a => a.state === 'I').forEach(a => drawAgent(a, scale, s));
    }

    function drawAgent(agent, scale, size) {
        const x = agent.x * scale;
        const y = agent.y * scale;
        const color = COLORS[agent.state] || COLORS.S;
        const baseRadius = Math.max(2, size / 120);
        let radius = baseRadius;

        // Glow effect for infected and exposed
        if (GLOW[agent.state]) {
            const glowRadius = radius * 3.5;
            const gradient = ctx.createRadialGradient(x, y, radius, x, y, glowRadius);
            gradient.addColorStop(0, GLOW[agent.state]);
            gradient.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.beginPath();
            ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            radius = baseRadius * 1.3;
        }

        // Dead agents are smaller and transparent
        if (agent.state === 'D') {
            radius = baseRadius * 0.6;
            ctx.globalAlpha = 0.4;
        }

        // Main dot
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Bright center for living agents
        if (agent.state !== 'D') {
            ctx.beginPath();
            ctx.arc(x, y, radius * 0.4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,0.4)';
            ctx.fill();
        }

        ctx.globalAlpha = 1.0;
    }

    return { init, setAgents, draw, resize };
})();
