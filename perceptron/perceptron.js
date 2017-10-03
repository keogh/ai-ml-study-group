function getRandom(interval) {
  return (Math.random()*interval) * (Math.floor(Math.random()*2) == 1 ? 1 : -1);
}
var perceptron = function () {
  return {
    weights: [],
    c: 0.01,
    bias: 1,

    init: function (numIn) {
      for (var i = 0; i < numIn+1; i++) {
        this.weights.push(getRandom(1));
      }
    },

    calculate: function (inputs) {
      var sum = 0;
      for (var i = 0; i < inputs.length; i++) {
        sum += inputs[i] * this.weights[i];
      }
      sum += this.bias * this.weights[this.weights.length - 1];
      return sum;
    },

    feedForward: function (inputs) {
      var sum = this.calculate(inputs);
      return this.activate(sum);
    },

    activate: function (sum) {
      return sum > 0 ? 1 : -1;
    },

    train: function (inputs, desired) {
      var guess = this.feedForward(inputs);
      var error = desired - guess;
      for (var i = 0; i < this.weights.length; i++) {
        this.weights[i] += this.c * error * inputs[i];
      }
    }
  }
}

var trainer = function (x, y, a) {
  return {
    inputs: [x, y, 1],
    answer: a
  }
}

var trainModel = function (trainData) {
  var ptron = perceptron();
  ptron.init(2);
  for (var i = 0; i < 20*trainData.length; i++) {
    ptron.train(trainData[i%trainData.length].inputs, trainData[i%trainData.length].answer);
  }
  return ptron;
}

function f(x) {
  return 2*x+1;
}

var generateTrainData = function (total) {
  var training = [];
  for (var i = 0; i < total; i++) {
    var x = getRandom(10);
    var y = getRandom(10);
    var answer = 1;
    if (y < f(x)) answer = -1;
    training[i] = trainer(x, y, answer);
    //console.log('x: ', x, ' y: ', y, 'a: ', answer);
  }
  return training;
}
