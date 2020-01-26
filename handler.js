const RADIUS = 18;
const SPACING = 68;
const HORIZONTAL_SPACING = 86;

const settings = [
    {
        canvasWidth: 600,
        canvasHeight: 240,
        layers: [3],
        network: '{"weights":[[[],[]],[[1,1],[1,1],[1,1]],[[1,1,1]]],"minWeight":1,"maxWeight":1,"activation":"tanh","learningRate":0.5}'
    },
    {
        canvasWidth: 600,
        canvasHeight: 240,
        layers: [3, 3],
        network: '{"weights":[[[],[]],[[1,1],[1,1],[1,1]],[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1]]],"minWeight":1,"maxWeight":1,"activation":"tanh","learningRate":0.5}'
    },
    {
        canvasWidth: 600,
        canvasHeight: 430,
        layers: [3, 6, 3],
        network: '{"weights":[[[],[]],[[1,1],[1,1],[1,1]],[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],[[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],[[1,1,1]]],"minWeight":1,"maxWeight":1,"activation":"tanh","learningRate":0.5}'
    },
    {
        canvasWidth: 700,
        canvasHeight: 430,
        layers: [3, 6, 6, 3],
        network: '{"weights":[[[],[]],[[1,1],[1,1],[1,1]],[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],[[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],[[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],[[1,1,1]]],"minWeight":1,"maxWeight":1,"activation":"tanh","learningRate":0.5}'
    }
];

function drawNeuron(x, y, context) {
    context.beginPath();
    context.arc(x, y, RADIUS, 0, 2 * Math.PI, false);
    context.fillStyle = 'white';
    context.fill();
    context.lineWidth = 1;
    context.strokeStyle = 'black';
    context.stroke();
}

function computeNeuronPositions(x, y, layerSize) {
    const size = layerSize * RADIUS * 2 + (layerSize - 1) * (SPACING - RADIUS * 2);
    y -= size / 2 - RADIUS;
    const positions = [];
    for (let i = 0; i < layerSize; ++i) {
        positions.push({ x, y: y + i * SPACING });
    }
    return positions;
}

function drawConnections(positions, context, weights, minWeight, maxWeight) {
    for (let i = positions.length - 2; i >= 0; --i) {
        for (let j = 0; j < positions[i].length; ++j) {
            for (let k = 0; k < positions[i + 1].length; ++k) {
                context.beginPath();
                const weight = weights[i + 1][k][j];
                context.lineWidth = 6 * (weight - minWeight) / (maxWeight - minWeight) + 1;
                context.moveTo(positions[i][j].x, positions[i][j].y);
                context.lineTo(positions[i + 1][k].x, positions[i + 1][k].y);
                context.stroke();
            }
        }
    }
}

function drawNetwork(x, y, context, { weights, minWeight, maxWeight }) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    const width = weights.length * RADIUS * 2 + (weights.length - 1) * (HORIZONTAL_SPACING - RADIUS * 2);
    x -= width / 2 - RADIUS;
    positions = weights.map((layer, index) => {
        return computeNeuronPositions(x + HORIZONTAL_SPACING * index, y, layer.length);
    });
    drawConnections(positions, context, weights, minWeight, maxWeight);
    for (let x = 0; x < weights.length; ++x) {
        for (let y = 0; y < weights[x].length; ++y) {
            drawNeuron(positions[x][y].x, positions[x][y].y, context);
        }
    }
    return positions;
}

function drawText(outputPosition, a, b, c, context) {
    context.fillStyle = 'black';
    context.font = '16px Arial';
    context.textBaseline = 'middle';
    context.fillText(`Input: ${b}`, positions[0][0].x - 90, positions[0][0].y);
    context.fillText(`Input: ${a}`, positions[0][1].x - 90, positions[0][1].y);
    context.fillText(`Output: ${c}`, outputPosition.x + 40, outputPosition.y);
}

const onMessage = ({ data }) => {
    const positions = drawNetwork(canvas.width / 2, canvas.height / 2, context, data.network);
    const outputPosition = positions[positions.length - 1][0];
    const display = document.getElementById(`${data.inputs[0]}${data.inputs[1]}`);
    const prediction = Math.round(data.output);
    drawText(outputPosition, data.inputs[0], data.inputs[1], prediction, context);
    display.innerHTML = prediction;
    display.className = data.inputs[0] + data.inputs[1] === prediction ? '' : 'error';
    epoch.innerHTML = `Epoch: ${String(data.iteration).padStart(7, '0')}`;
    if (data.iteration % 300 === 0) {
        chart.data.labels.push(data.iteration);
        chart.data.datasets[0].data.push(data.error);
        if (chart.data.labels.length > 15) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update();
    }
};

const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const graph = document.getElementById('graph');
const graphContext = graph.getContext('2d')
const epoch = document.getElementById('epoch');
const start = document.getElementById('start');
const layerSelect = document.getElementById('layers');
const rate = document.getElementById('rate');
const result = document.getElementById('result');
const activationSelect = document.getElementById('activation');

const chart = new Chart(graphContext, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'Average Error',
                backgroundColor: 'rgb(255, 99, 132)',
                borderColor: 'rgb(255, 99, 132)',
                data: []
            }
        ]
    }
});

chart.canvas.parentNode.style.height = '300px';
chart.canvas.parentNode.style.width = '600px';
canvas.height = settings[0].canvasHeight;
canvas.width = settings[0].canvasWidth;

let isTraining = false;
let worker = null;

start.addEventListener('click', () => {
    if (isTraining) {
        worker.terminate();
        isTraining = false;
        start.innerHTML = 'Start';
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
    } else {
        if (window.Worker) {
            for (let i = 0; i <= 9; ++i) {
                for (let j = 0; j <= 9; ++j) {
                    if (i + j < 10) {
                        const element = document.getElementById(`${i}${j}`);
                        element.innerHTML = 'NaN';
                        element.className = 'error';
                    }
                }
            }
            const index = +layerSelect.value;
            canvas.height = settings[index].canvasHeight;
            canvas.width = settings[index].canvasWidth;
            worker = new Worker('worker.js');
            worker.onmessage = onMessage;
            start.innerHTML = 'Stop';
            isTraining = true;
            worker.postMessage({
                activation: activationSelect.value,
                layers: settings[index].layers,
                learningRate: +rate.value
            });
        } else {
            alert('Sorry, this visualization is not supported by your browser.');
        }
    }
});

layerSelect.addEventListener('change', () => {
    if (!isTraining) {
        const index = +layerSelect.value;
        canvas.height = settings[index].canvasHeight;
        canvas.width = settings[index].canvasWidth;
        const positions = drawNetwork(canvas.width / 2, canvas.height / 2, context, JSON.parse(settings[index].network));
        const outputPosition = positions[positions.length - 1][0];
        drawText(outputPosition, 0, 0, 0, context);
    }
});

const initial = drawNetwork(canvas.width / 2, canvas.height / 2, context, JSON.parse(settings[1].network));
const initialOutput = initial[initial.length - 1][0];
drawText(initialOutput, 0, 0, 0, context);

for (let i = 0; i <= 9; ++i) {
    for (let j = 0; j <= 9; ++j) {
        if (i + j < 10) {
            result.innerHTML += `<p>${i} + ${j} = <span class="error" id=${i}${j}>NaN</span></p>`;
        }
    }
}
