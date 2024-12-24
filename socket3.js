const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { Image, createCanvas } = require('canvas');
const tf = require('@tensorflow/tfjs-node-gpu');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);
const canvas = createCanvas(640, 480);
const context = canvas.getContext('2d');

let model = null;

// Serve the index.html file
app.get('/', function (req, res) {
    res.sendFile(__dirname + '/public/index2.html');
});

// 모델 로드
async function loadModel() {
  const modelPath = 'file://./yolomodel/model.json';
  model = await tf.loadGraphModel(modelPath);
  console.log('Model loaded.');
}

// Load the model before starting the server
loadModel().then(() => {
    server.listen(3000, function () {
        console.log('Socket IO server listening on port 3000');
    });
});

// 이미지 예측 처리
async function predictImage(imageData) {
  const image = new Image();
  image.src = imageData;
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const inputTensor = tf.tidy(() => {
    return tf.image.resizeBilinear(tf.browser.fromPixels(canvas), [640, 640])
      .div(255.0)
      .expandDims(0);
  });

  const [boxes, scores, classes, valid_detections] = await model.executeAsync(inputTensor);
  tf.dispose(inputTensor);

  return {
    boxes: Array.from(boxes.dataSync()),
    scores: Array.from(scores.dataSync()),
    classes: Array.from(classes.dataSync()),
    valid_detections: Array.from(valid_detections.dataSync())[0]
  };
}

// Socket.io 이벤트 처리
io.on('connection', (socket) => {
  console.log('Client connected');

  socket.on('upload', async (file, callback) => {
    const result = await predictImage(file.Data);
    socket.emit('isfinish', [result.boxes, result.scores, result.classes, result.valid_detections]);
    callback('Processing completed.');
  });
});