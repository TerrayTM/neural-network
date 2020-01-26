self.onmessage = ({ data }) => {
    importScripts('network.js');

    const instances = [];
    const network = new Network(2, 1, data.layers, { Tanh: tanh, Sigmoid: sigmoid }[data.activation], data.learningRate);
    const iterations = 9999999;
    let count = 0;

    for (let i = 0; i <= 9; ++i) {
        for (let j = 0; j <= 9; ++j) {
           if (i + j < 10) {
                instances.push([i / 9, j / 9, (i + j) / 9]);
           }
        }
    }

    function train() {
        const i = count % instances.length;
        network.setInputs([instances[i][0], instances[i][1]]);
        const [sum] = network.feedForward();
        network.backpropagation([instances[i][2]]);
        const averageError = network.computeError([instances[i][2]]);
        postMessage({
            network: network.export(),
            inputs: [instances[i][0] * 9, instances[i][1] * 9],
            output: sum * 9,
            iteration: count,
            error: averageError
        });
        ++count;
        if (count <= iterations) {
            setTimeout(train, 3);
        }
    }

    train();
};
