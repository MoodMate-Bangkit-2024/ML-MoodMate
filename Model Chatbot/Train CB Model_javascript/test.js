const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;

const MODEL_PATH = './tfjs_model/model.json';  
const WORD_INDEX_PATH = './word_index.json';   
const DATA_PATH = './data.json';               

let model;
let wordIndex;
let intents;

async function loadModel() {
    model = await tf.loadLayersModel(`file://${MODEL_PATH}`);
    console.log('Model loaded');
}

async function loadWordIndex() {
    const wordIndexData = await fs.readFile(WORD_INDEX_PATH, 'utf8');
    wordIndex = JSON.parse(wordIndexData);
    console.log('Word index loaded');
}

async function loadIntents() {
    const intentsData = await fs.readFile(DATA_PATH, 'utf8');
    intents = JSON.parse(intentsData).intents;
    console.log('Intents loaded');
}

// predict
async function predictResponse(userInput) {
    const preprocessedInput = removePunctuationAndLowercase(userInput).split(' ').map(word => wordIndex[word] || 0);
    const maxLen = 12;
    const paddedInput = padSequences([preprocessedInput], maxLen);
    const tensorInput = tf.tensor2d(paddedInput, [1, maxLen]);

    const prediction = model.predict(tensorInput);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    const predictedTag = intents[predictedIndex].tag;

    // response for predicted tag
    const response = intents.find(intent => intent.tag === predictedTag).responses[0];
    return response;
}

// preprocess
function removePunctuationAndLowercase(text) {
    return text.toLowerCase().replace(/[^\w\s]/gi, '');
}

function padSequences(sequences, maxLen) {
    return sequences.map(seq => {
        const padding = new Array(maxLen - seq.length).fill(0);
        return padding.concat(seq).slice(-maxLen);
    });
}

// function to test the model
async function main() {
    await loadModel();
    await loadWordIndex();
    await loadIntents();

    // Test with user input
    const userInput = 'Halo!';
    const response = await predictResponse(userInput);
    console.log(`User Input: ${userInput}`);
    console.log(`Model Response: ${response}`);
}

// Run the main function
main().catch(err => console.error(err));
