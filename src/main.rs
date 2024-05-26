use rand::Rng; // Import the random number generator from the rand crate

// Define constants for the neural network structure
const NUM_INPUTS: usize = 2; // Number of input neurons
const NUM_HIDDEN: usize = 2; // Number of hidden neurons
const NUM_OUTPUTS: usize = 1; // Number of output neurons
const NUM_TRAINING_SETS: usize = 4; // Number of training examples

const EPOCHS: i32 = 100000; // Number of training iterations

/// Initializes weights randomly for the given weight matrix.
///
/// # Arguments
///
/// * `weights` - A mutable reference to a vector of vectors representing the weight matrix.
fn initialize_weights(weights: &mut Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng(); // Create a random number generator
    for row in weights.iter_mut() {
        // Iterate through each row of weights
        for weight in row.iter_mut() {
            // Iterate through each weight in the row
            *weight = rng.gen_range(0.0..1.0); // Assign a random value between 0.0 and 1.0
        }
    }
}

/// Sigmoid activation function.
///
/// # Arguments
///
/// * `x` - A floating point number.
///
/// # Returns
///
/// * A floating point number representing the sigmoid of x.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp()) // Compute the sigmoid of x
}

/// Derivative of the sigmoid function.
///
/// # Arguments
///
/// * `x` - A floating point number.
///
/// # Returns
///
/// * A floating point number representing the derivative of sigmoid for backpropagation.
fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x) // Compute the derivative of sigmoid for backpropagation
}

/// Calculates the accuracy of the neural network.
///
/// # Arguments
///
/// * `training_data` - A vector of vectors representing the input training data.
/// * `training_output` - A vector of vectors representing the expected output data.
/// * `hidden_weights` - A vector of vectors representing the weights between the input and hidden layers.
/// * `output_weights` - A vector of vectors representing the weights between the hidden and output layers.
/// * `hidden_layer_bias` - A vector representing the biases for the hidden layer neurons.
/// * `output_layer_bias` - A vector representing the biases for the output layer neurons.
///
/// # Returns
///
/// * A floating point number representing the accuracy of the network.
fn calculate_accuracy(
    training_data: &Vec<Vec<f32>>,
    training_output: &Vec<Vec<f32>>,
    hidden_weights: &Vec<Vec<f32>>,
    output_weights: &Vec<Vec<f32>>,
    hidden_layer_bias: &Vec<f32>,
    output_layer_bias: &Vec<f32>,
) -> f32 {
    let mut correct_count = 0; // Counter for correct predictions

    for i in 0..NUM_TRAINING_SETS {
        let mut hidden_layer = vec![0.0; NUM_HIDDEN]; // Initialize hidden layer activations
        let mut output_layer = vec![0.0; NUM_OUTPUTS]; // Initialize output layer activations

        // FORWARD PASS
        // COMPUTE HIDDEN LAYER ACTIVATION
        for j in 0..NUM_HIDDEN {
            let mut activation = hidden_layer_bias[j]; // Start with the bias
            for k in 0..NUM_INPUTS {
                activation += training_data[i][k] * hidden_weights[j][k]; // Add weighted input
            }
            hidden_layer[j] = sigmoid(activation); // Apply sigmoid function
        }

        // COMPUTE OUTPUT LAYER ACTIVATION
        for j in 0..NUM_OUTPUTS {
            let mut activation = output_layer_bias[j]; // Start with the bias
            for k in 0..NUM_HIDDEN {
                activation += hidden_layer[k] * output_weights[j][k]; // Add weighted hidden layer activation
            }
            output_layer[j] = sigmoid(activation); // Apply sigmoid function
        }

        // PREDICT
        let prediction = output_layer[0].round(); // Rounding to 0 or 1 for binary classification
        if prediction == training_output[i][0] {
            correct_count += 1; // Increment correct count if prediction matches expected output
        }
    }

    correct_count as f32 / NUM_TRAINING_SETS as f32 // Calculate accuracy as a fraction
}

/// The main function which sets up and trains the neural network.
fn main() {
    let learning_rate = 0.1; // Learning rate for weight updates

    // Initialize layer activations
    let mut hidden_layer = vec![0.0; NUM_HIDDEN];
    let mut output_layer = vec![0.0; NUM_OUTPUTS];

    // Initialize biases for each layer
    let mut hidden_layer_bias = vec![0.0; NUM_HIDDEN];
    let mut output_layer_bias = vec![0.0; NUM_OUTPUTS];

    // Initialize weight matrices
    let mut hidden_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_INPUTS]; NUM_HIDDEN];
    let mut output_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_HIDDEN]; NUM_OUTPUTS];

    // Training data and expected outputs for XOR problem
    let training_data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], // Input: 0 0
        vec![0.0, 1.0], // Input: 0 1
        vec![1.0, 0.0], // Input: 1 0
        vec![1.0, 1.0], // Input: 1 1
    ];

    let training_output: Vec<Vec<f32>> = vec![
        // XOR outputs
        vec![0.0], // Output: 0
        vec![1.0], // Output: 1
        vec![1.0], // Output: 1
        vec![0.0], // Output: 0
    ];

    // INITIALIZE WEIGHTS
    initialize_weights(&mut hidden_weights); // Initialize hidden layer weights
    initialize_weights(&mut output_weights); // Initialize output layer weights

    // INITIALIZE BIAS
    let mut rng = rand::thread_rng();
    for i in 0..NUM_HIDDEN {
        hidden_layer_bias[i] = rng.gen_range(0.0..1.0); // Randomize hidden layer biases
    }
    for i in 0..NUM_OUTPUTS {
        output_layer_bias[i] = rng.gen_range(0.0..1.0); // Randomize output layer biases
    }

    let training_set_order = vec![0, 1, 2, 3]; // Order of training examples

    // TRAIN THE NETWORK
    for _epoch in 0..EPOCHS {
        for &i in &training_set_order {
            // FORWARD PASS

            // COMPUTE HIDDEN LAYER ACTIVATION
            for j in 0..NUM_HIDDEN {
                let mut activation = hidden_layer_bias[j]; // Start with the bias
                for k in 0..NUM_INPUTS {
                    activation += training_data[i][k] * hidden_weights[j][k]; // Add weighted input
                }
                hidden_layer[j] = sigmoid(activation); // Apply sigmoid function
            }

            // COMPUTE OUTPUT LAYER ACTIVATION
            for j in 0..NUM_OUTPUTS {
                let mut activation = output_layer_bias[j]; // Start with the bias
                for k in 0..NUM_HIDDEN {
                    activation += hidden_layer[k] * output_weights[j][k]; // Add weighted hidden layer activation
                }
                output_layer[j] = sigmoid(activation); // Apply sigmoid function
            }
            println!(
                "Input: {:?} -> Predicted output: {:?} (Expected: {:?})",
                training_data[i], output_layer, training_output[i]
            );

            // BACKPROPAGATION

            // COMPUTE CHANGE IN OUTPUT WEIGHTS
            let mut delta_output = vec![0.0; NUM_OUTPUTS];
            for b in 0..NUM_OUTPUTS {
                let error = training_output[i][b] - output_layer[b]; // Calculate error
                delta_output[b] = error * sigmoid_derivative(output_layer[b]); // Calculate delta for output layer
            }

            let mut delta_hidden = vec![0.0; NUM_HIDDEN];
            for m in 0..NUM_HIDDEN {
                let mut new_error = 0.0;
                for v in 0..NUM_OUTPUTS {
                    new_error += delta_output[v] * output_weights[v][m]; // Propagate error back to hidden layer
                }
                delta_hidden[m] = new_error * sigmoid_derivative(hidden_layer[m]);
                // Calculate delta for hidden layer
            }

            // APPLY CHANGE IN OUTPUT WEIGHTS
            for c in 0..NUM_OUTPUTS {
                output_layer_bias[c] += delta_output[c] * learning_rate; // Update output layer bias
                for z in 0..NUM_HIDDEN {
                    output_weights[c][z] += hidden_layer[z] * delta_output[c] * learning_rate;
                    // Update output layer weights
                }
            }

            // APPLY CHANGE IN HIDDEN WEIGHTS
            for p in 0..NUM_HIDDEN {
                hidden_layer_bias[p] += delta_hidden[p] * learning_rate; // Update hidden layer bias
                for o in 0..NUM_INPUTS {
                    hidden_weights[p][o] += training_data[i][o] * delta_hidden[p] * learning_rate;
                    // Update hidden layer weights
                }
            }
        }
    }

    // Calculate and print final accuracy
    let accuracy = calculate_accuracy(
        &training_data,
        &training_output,
        &hidden_weights,
        &output_weights,
        &hidden_layer_bias,
        &output_layer_bias,
    );
    println!("Final accuracy: {:.2}%", accuracy * 100.0);

    // Print final weights and biases
    println!("Final hidden weights: {:?}", hidden_weights);
    println!("Final output weights: {:?}", output_weights);
    println!("Final hidden biases: {:?}", hidden_layer_bias);
    println!("Final output biases: {:?}", output_layer_bias);
}
