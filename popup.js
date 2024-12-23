// Load the Logistic Regression model and pipeline components
fetch('logistic_regression_model.json')
  .then(response => response.json())
  .then(model => {
    const { coefficients, intercept, classes, vocabulary, idf } = model;

    // Preprocess input review
    function preprocess(review) {
      const tokens = review.toLowerCase().split(/\W+/);
      const tf = {};
      tokens.forEach(token => {
        if (vocabulary[token] !== undefined) {
          tf[token] = (tf[token] || 0) + 1;
        }
      });

      // Compute TF-IDF vector
      const vector = Array(Object.keys(vocabulary).length).fill(0);
      for (const [word, freq] of Object.entries(tf)) {
        const idx = vocabulary[word];
        vector[idx] = freq * idf[idx];
      }
      return vector;
    }

    // Predict function
    function predict(review) {
      const features = preprocess(review);
      const linearCombination = coefficients[0].reduce(
        (sum, coeff, idx) => sum + coeff * features[idx],
        intercept[0]
      );

      // Logistic function
      const probability = 1 / (1 + Math.exp(-linearCombination));
      return probability >= 0.5 ? classes[1] : classes[0];
    }

    // Handle button click
    document.getElementById('predict-btn').addEventListener('click', () => {
      const review = document.getElementById('review-input').value;
      const result = predict(review);
      document.getElementById('result').innerText = `Prediction: ${result}`;
    });
  });
