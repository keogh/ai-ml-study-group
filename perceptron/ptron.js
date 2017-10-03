const EPOCH = 1500, EPOCH_END = 4000, TRAINING = 1, TRANSITION = 2, SHOW = 3, STOP = 4;

var ptron;
var counter = 0;
var learnRate = 0.02;
var state = TRAINING;

function setup() {
    createCanvas( 800, 600 );
    clearBack();
    ptron = perceptron();
    ptron.init(2);
}

function draw() {
    switch( state ) {
        case TRAINING: training(); break;
        case TRANSITION: transition(); break;
        case SHOW: show(); break;
    }
}

function clearBack() {
    background( 0 );
    stroke( 255 );
    strokeWeight( 4 );

    var x = width;
    line( 0, 0, x, lineDef( x ) );
}

function transition() {
    clearBack();
    state = SHOW;
}

function lineDef( x ) {
    return 0.75 * x;
}

function training() {
    var a = random( width ),
        b = random( height );

    lDef = lineDef( a ) > b ? -1 : 1;

    ptron.setInput( [a, b] );
    ptron.feedForward([a, b]);
    var pRes = ptron.getOutput();
    var match = (pRes == lDef);
    var clr;

    if( !match ) {
        var err = ( pRes - lDef ) * learnRate;
        ptron.adjustWeights( err );

        clr = color( 255, 0, 0 );

    } else {
        clr = color( 0, 255, 0 );
    }

    noStroke();
    fill( clr );
    ellipse( a, b, 4, 4 );

    if( ++counter == EPOCH ) state = TRANSITION;
}

function show() {
    var a = random( width ),
        b = random( height ),
        clr;

    ptron.setInput( [a, b] );
    ptron.feedForward();
    var pRes = ptron.getOutput();

    if( pRes < 0 )
        clr = color( 255, 0, 0 );
    else
        clr = color( 0, 255, 0 );

    noStroke();
    fill( clr );
    ellipse( a, b, 4, 4 );

    if ( ++counter == EPOCH_END) state = STOP;
}

function stop() {
  return;
}

function getRandom(interval) {
  return (Math.random()*interval) * (Math.floor(Math.random()*2) == 1 ? 1 : -1);
}
var perceptron = function () {
  return {
    weights: [],
    input: [],
    output: null,
    c: 0.01,
    bias: 1,

    init: function (numIn) {
      for (var i = 0; i < numIn+1; i++) {
        this.weights.push(Math.random());
      }
    },

    setInput: function (input) {
      this.input = input;
    },

    getOutput: function () {
      return this.output;
    },

    feedForward: function () {
      var sum = 0;
      for (var i = 0; i < this.input.length; i++) {
        sum += this.input[i] * this.weights[i];
      }
      sum += this.bias * this.weights[this.weights.length - 1];
      this.output = this.activate(sum);
    },

    activate: function (sum) {
      //return sum > 0 ? 1 : -1;
      return( Math.tanh( sum ) < .5 ? 1 : -1 );
    },

    adjustWeights: function (err) {
      for (var i = 0; i < this.weights.length; i++) {
        this.weights[i] += err * (this.input[i] || this.bias);
      }
    }
  }
}
