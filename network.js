//--------------------------------------------------------
//| Simple neural networks implementation by Terry Zheng |
//--------------------------------------------------------

const sigmoid = {
  activation: input => 1 / (1 + Math.exp(-input)),
  derivative: input => input * (1 - input)
};

const tanh = {
  activation: input => Math.tanh(input),
  derivative: input => 1 - input * input
};

class Neuron {
  constructor(inputs, learningRate, { activation, derivative }) {
    this._weights = new Array(inputs).fill(Math.random());
    this._bias = Math.random();
    this._value = null;
    this._activation = activation;
    this._derivative = derivative;
    this._learningRate = learningRate;
  }

  computeOutput(inputs) {
    this._value = this._activation(this.weightedSum(inputs));
  }

  weightedSum(inputs) {
    return inputs.reduce((previous, next, index) => previous + next * this._weights[index], this._bias);
  }

  computeError(expected) {
    const difference = expected - this._value;
    return 0.5 * difference * difference;
  }

  setValue(value) {
    this._value = value;
  }

  info() {
    return `(w=[${this._weights.map(weight => weight.toFixed(3)).join(',')}], b=${this._bias.toFixed(3)}, v=${this._value.toFixed(3)})`;
  }

  computeBackpropagation(inputs, expected, isInputLayer) {
    const partials = [];
    for (let i = 0; i < this._weights.length; ++i) {
      let delta = this._derivative(this._value);
      delta *= isInputLayer ? -(expected - this._value) : expected;
      const gradient = delta * inputs[i];
      partials.push(delta * this._weights[i]);
      this._weights[i] -= this._learningRate * gradient;
      this._bias -= this._learningRate / 10 * delta;
    }
    return partials;
  }

  get value() {
    return this._value;
  }
}

class Layer {
  constructor(size, inputsPerNeuron, activation, learningRate) {
    this._neurons = [];
    for (let i = 0; i < size; ++i) {
      this._neurons.push(new Neuron(inputsPerNeuron, learningRate, activation));
    }
  }

  computeOutputs(inputs) {
    return this._neurons.forEach(neuron => neuron.computeOutput(inputs));
  }

  setInputs(inputs) {
    this._neurons.forEach((neuron, index) => neuron.setValue(inputs[index]));
  }

  getOutputs() {
    return this._neurons.map(neuron => neuron.value);
  }

  computeError(expected) {
    return this._neurons.map((neuron, index) => neuron.computeError(expected[index]));
  }

  info() {
    return this._neurons.map(neuron => neuron.info()).join(' ');
  }

  computeBackpropagation(inputs, expected, isInputLayer) {
    const partials = this._neurons.map((neuron, index) => neuron.computeBackpropagation(inputs, expected[index], isInputLayer));
    const results = new Array(inputs.length).fill(0);
    for (let i = 0; i < partials.length; ++i) {
      for (let j = 0; j < partials[i].length; ++j) {
        results[j] += partials[i][j];
      }
    }
    return results;
  }
}

class Network {
  constructor(inputs, outputs, hiddenLayers, activation, learningRate) {
    this._layers = [new Layer(inputs, 0, activation, learningRate)];
    for (let i = 0; i < hiddenLayers.length; ++i) {
      this._layers.push(new Layer(hiddenLayers[i], i == 0 ? inputs : hiddenLayers[i - 1], activation, learningRate));
    }
    this._layers.push(new Layer(outputs, hiddenLayers[hiddenLayers.length - 1], activation, learningRate));
    this._inputLayer = this._layers[0];
    this._outputLayer = this._layers[this._layers.length - 1];
  }

  feedForward() {
    for (let i = 1; i < this._layers.length; ++i) {
      this._layers[i].computeOutputs(this._layers[i - 1].getOutputs());
    }
  }

  setInputs(inputs) {
    this._inputLayer.setInputs(inputs);
  }

  computeError(expected) {
    return this._outputLayer.computeError(expected).reduce((previous, next) => previous + next, 0);
  }

  info() {
    return this._layers.map(layer => layer.info()).join('\n');
  }

  backpropagation(expected) {
    for (let i = this._layers.length - 1; i >= 1; --i) {
      expected = this._layers[i].computeBackpropagation(this._layers[i - 1].getOutputs(), expected, i === this._layers.length - 1);
    }
  }

  predict(inputs) {
    this.setInputs(inputs);
    this.feedForward();
    return this._outputLayer.getOutputs();
  }
}
