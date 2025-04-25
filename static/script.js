let vertices = [];
let canvas, ctx;
let loadingBarInterval = null;
let progress = 1;

window.onload = () => {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    resizeCanvas();

    canvas.addEventListener('click', onCanvasClick);
    document.getElementById('simulateBtn').addEventListener('click', simulateShape);
    document.getElementById('clearBtn').addEventListener('click', clearCanvas);
    document.getElementById('optimiseBtn').addEventListener('click', optimiseShape);
};

function startLoadingBar() {
    const bar = document.getElementById('loadingBar');
    const container = document.getElementById('loadingBarContainer');

    progress = 1;
    bar.style.width = progress + '%';
    container.style.display = 'block';

    loadingBarInterval = setInterval(() => {
        progress = progress + (100 - progress) * 0.033;  // asymptotic to 100%
        bar.style.width = progress + '%';
    }, 500);  // update every 0.5s
}

function stopLoadingBar() {
    clearInterval(loadingBarInterval);
    loadingBarInterval = null;
    document.getElementById('loadingBarContainer').style.display = 'none';
}


function clearCanvas() {
    vertices = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function resizeCanvas() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
}

function onCanvasClick(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    vertices.push([x, y]);
    drawPolygon();
}

function drawPolygon() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (vertices.length === 0) return;

    ctx.beginPath();
    const [startX, startY] = [vertices[0][0] * canvas.width, vertices[0][1] * canvas.height];
    ctx.moveTo(startX, startY);

    for (let [x, y] of vertices.slice(1)) {
        ctx.lineTo(x * canvas.width, y * canvas.height);
    }
    ctx.closePath();

    // Fill with a cool translucent blue
    ctx.fillStyle = 'rgba(50, 150, 255, 0.4)';
    ctx.fill();

    // Stroke the outline
    ctx.strokeStyle = '#003366';
    ctx.lineWidth = 2;
    ctx.stroke();
}


function scaleVerticesTo128(vertices, width, height) {
    // Step 1: Scale and flip Y
    let scaled = vertices.map(([x, y]) => [
        x * 128,
        (1 - y) * 128
    ]);

    console.log('Scaled vertices (before centering):', scaled);

    // Step 2: Compute centroid
    let sumX = 0, sumY = 0;
    scaled.forEach(([x, y]) => {
        sumX += x;
        sumY += y;
    });
    const n = scaled.length;
    const centroidX = sumX / n;
    const centroidY = sumY / n;

    console.log('Centroid before shifting:', [centroidX, centroidY]);

    // Step 3: Translate to center at (64, 64)
    const dx = 64 - centroidX;
    const dy = 64 - centroidY;

    console.log('Translation offset (dx, dy):', [dx, dy]);

    const centered = scaled.map(([x, y]) => [
        Math.round(x + dx),
        Math.round(y + dy)
    ]);

    console.log('Centered vertices (final result):', centered);

    return centered;
}


async function simulateShape() {
    const loadingMsg = document.getElementById('loadingMsg');
    loadingMsg.style.display = 'block';

    const scaledVerts = scaleVerticesTo128(vertices, canvas.width, canvas.height);

    try {
        const response = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vertices: scaledVerts })
        });
        const data = await response.json();

        // Update image
        document.getElementById('cfdImage').src = `data:image/png;base64,${data.image}`;

        // Update table
        const tbody = document.getElementById('resultsTable').querySelector('tbody');
        tbody.innerHTML = '';
        data.results.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.name}</td>
                <td>${row.surrogate}</td>
                <td>${row.cfd}</td>
                <td><input type="text" id="target-${row.name}" placeholder="e.g. 0.5"/></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.error('Simulation error:', err);
        alert('Something went wrong with the simulation.');
    } finally {
        loadingMsg.style.display = 'none';
    }
}

async function optimiseShape() {
    const loadingMsg = document.getElementById('loadingMsg');
    loadingMsg.style.display = 'block';
    startLoadingBar();


    const inputs = document.querySelectorAll('input[id^="target-"]');
    const targets = {};
    inputs.forEach(input => {
        const varName = input.id.replace('target-', '');
        const val = parseFloat(input.value);
        if (!isNaN(val)) targets[varName] = val;
    });

    try {
        const response = await fetch('/optimise', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ targets: targets })
        });

        let text = await response.text();  // get raw text
        console.log('Raw optimise response:', text);

        try {
            const data = JSON.parse(text);
            console.log('Parsed JSON:', data);

            if (!Array.isArray(data.vertices)) {
                throw new Error("No 'vertices' array in response.");
            }

            // vertices = data.vertices;
            vertices = data.vertices.map(([x, y]) => [
                x / 128,
                1 - (y / 128)  // flip y back
            ]);
            drawPolygon();
            simulateShape();

        } catch (parseError) {
            console.error('Error parsing optimise response JSON:', parseError);
            alert('Something went wrong parsing the optimisation response.');
        }


    } catch (err) {
        console.error('Optimisation error:', err);
        alert('Something went wrong with optimisation.');
    } finally {
        stopLoadingBar();
        loadingMsg.style.display = 'none';
    }
}
