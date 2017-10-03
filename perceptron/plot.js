(function (trainModel, generateTrainData) {
  var fn = '';
  var $form;
  var dots = [];
  window.ptron = null;

  var point = function (x, y, guess) {
    return {x: x, y: y, guess: guess}
  };

  window.onload = function () {
    $form = document.forms.controls;
    plot();

    $form['btn-guess'].onclick = function () {
      var x = $form.x.value * 1;
      var y = $form.y.value * 1;
      if (!isNaN(x) && !isNaN(y) && ptron) {
        var guess = ptron.feedForward([x, y, 1]);
        dots.push(point(x, y, guess));
        plot();
      }
    }

    $form['btn-random-input'].onclick = function () {
      dots = [];
      var total = 3000;
      var range = 3;
      for (var i = 0; i < total; i++) {
        var x = getRandom(range);
        var y = getRandom(range);
        var guess = ptron.feedForward([x, y]);
        dots.push(point(x, y, guess));
      }
      plot();
    }

    var trainData = generateTrainData(1000000);
    ptron = trainModel(trainData);
  }

  function plot() {
    fn = $form.equation.value;
    var data = [{ fn: fn }];
    var pointsRed = [];
    var pointsBlue = [];

    for (var i = 0; i < dots.length; i++) {
      if (dots[i].guess === -1) {
        pointsRed.push([dots[i].x, dots[i].y]);
      } else {
        pointsBlue.push([dots[i].x, dots[i].y]);
      }
    }

    if (pointsRed.length) {
      data.push({
        points: pointsRed,
        fnType: 'points',
        graphType: 'scatter',
        color: 'red'
      });
    }
    if (pointsBlue.length) {
      data.push({
        points: pointsBlue,
        fnType: 'points',
        graphType: 'scatter',
        color: 'blue'
      });
    }

    if (ptron) {
      data.push({
        fnType: 'points',
        graphType: 'polyline',
        points: [
          [10, getY(10)],
          [-10, getY(-10)]
        ],
        color: 'orange'
      });
    }

    functionPlot({
      target: '#plot',
      data: data
    });
  }

  function getY(x) {
    return (-ptron.weights[0]*x-ptron.weights[2])/ptron.weights[1];
  }
})(trainModel, generateTrainData);
