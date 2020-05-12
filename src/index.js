import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom';
import * as serviceWorker from './serviceWorker';

import style from './index.module.css'

import * as tf from '@tensorflow/tfjs';

const OOV_CHAR = 2;
const PAD_CHAR = 0;

function App() {

  const [model, setModel] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [text, setText] = useState(null);

  async function fetchModel() {

    try {
      const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json')
      const metadataJson = await fetch('https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json');
      const sentimentMetadata = await metadataJson.json();

      setMetadata(sentimentMetadata)
      setModel(model)

    } catch (e) {
      setErrorMessage(e.message)
    }
  }

  // code highly inspired from : https://github.com/ml5js/ml5-library
  function padSequences(sequences, maxLen, padding = 'pre', truncating = 'pre', value = PAD_CHAR) {
    return sequences.map((seq) => {
      // Perform truncation.
      if (seq.length > maxLen) {
        if (truncating === 'pre') {
          seq.splice(0, seq.length - maxLen);
        } else {
          seq.splice(maxLen, seq.length - maxLen);
        }
      }
      // Perform padding.
      if (seq.length < maxLen) {
        const pad = [];
        for (let i = 0; i < maxLen - seq.length; i += 1) {
          pad.push(value);
        }
        if (padding === 'pre') {
          // eslint-disable-next-line no-param-reassign
          seq = pad.concat(seq);
        } else {
          // eslint-disable-next-line no-param-reassign
          seq = seq.concat(pad);
        }
      }
      return seq;
    });
  }

  function predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
      text.trim().toLowerCase().replace(/[.,?!]/g, '').split(' ');
    // Convert the words to a sequence of word indices.

    const sequence = inputText.map((word) => {
      let wordIndex = metadata.word_index[word] + metadata.index_from;
      if (wordIndex > metadata.vocabulary_size) {
        wordIndex = OOV_CHAR;
      }
      return wordIndex;
    });

    // Perform truncation and padding.
    const paddedSequence = padSequences([sequence], metadata.max_len);
    const input = tf.tensor2d(paddedSequence, [1, metadata.max_len]);
    const predictOut = model.predict(input);
    const score = predictOut.dataSync()[0];
    predictOut.dispose();
    input.dispose();
    console.log(score)
    setPrediction(score)
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (e.target.value.length < metadata.max_len) {
      setText(e.target.value)
      if (text) {
        predict(text)
      }
    }
  }

  useEffect(() => {

    fetchModel()

  }, []);

  // tbd if best way to catch error in React
  if (errorMessage) {
    return (
      <div>Error while fetching model, {errorMessage}</div>
    )
  }


  return (
    model ?

      <div className={style.ai}>
        <p>Type a sentence to get sentiment:</p>
        <form>
          <input type="text" onChange={handleSubmit} value={text ? text : ''} />
          <button type="submit" onClick={handleSubmit}>submit</button>
        </form>

        <div className={style.bar} >
          <div className={style.progress} style={{ width: (prediction * 100) + "%" }} >
          </div>
        </div>

      </div>

      : <div>"Loading model, hold on!"</div>
  )

}


ReactDOM.render(
  <React.StrictMode>
    <App ></App>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
