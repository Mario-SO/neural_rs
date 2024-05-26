use rand::Rng;

const NUM_INPUTS: usize = 2;
const NUM_HIDDEN: usize = 2;
const NUM_OUTPUTS: usize = 1;
const NUM_TRAINING_SETS: usize = 4;

const EPOCHS: i32 = 100000;

fn initialize_weights(weights: &mut Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    for row in weights.iter_mut() {
        for weight in row.iter_mut() {
            *weight = rng.gen_range(0.0..1.0);
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

fn calculate_accuracy(
    training_data: &Vec<Vec<f32>>,
    training_output: &Vec<Vec<f32>>,
    hidden_weights: &Vec<Vec<f32>>,
    output_weights: &Vec<Vec<f32>>,
    hidden_layer_bias: &Vec<f32>,
    output_layer_bias: &Vec<f32>,
) -> f32 {
    let mut correct_count = 0;

    for i in 0..NUM_TRAINING_SETS {
        let mut hidden_layer = vec![0.0; NUM_HIDDEN];
        let mut output_layer = vec![0.0; NUM_OUTPUTS];

        // FORWARD PASS
        // COMPUTE HIDDEN LAYER ACTIVATION
        for j in 0..NUM_HIDDEN {
            let mut activation = hidden_layer_bias[j];
            for k in 0..NUM_INPUTS {
                activation += training_data[i][k] * hidden_weights[j][k];
            }
            hidden_layer[j] = sigmoid(activation);
        }

        // COMPUTE OUTPUT LAYER ACTIVATION
        for j in 0..NUM_OUTPUTS {
            let mut activation = output_layer_bias[j];
            for k in 0..NUM_HIDDEN {
                activation += hidden_layer[k] * output_weights[j][k];
            }
            output_layer[j] = sigmoid(activation);
        }

        // PREDICT
        let prediction = output_layer[0].round(); // rounding to 0 or 1 for binary classification
        if prediction == training_output[i][0] {
            correct_count += 1;
        }
    }

    correct_count as f32 / NUM_TRAINING_SETS as f32
}

fn main() {
    let learning_rate = 0.1;

    let mut hidden_layer = vec![0.0; NUM_HIDDEN];
    let mut output_layer = vec![0.0; NUM_OUTPUTS];

    let mut hidden_layer_bias = vec![0.0; NUM_HIDDEN];
    let mut output_layer_bias = vec![0.0; NUM_OUTPUTS];

    let mut hidden_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_INPUTS]; NUM_HIDDEN];
    let mut output_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_HIDDEN]; NUM_OUTPUTS];

    let training_data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let training_output: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // INITIALIZE WEIGHTS
    initialize_weights(&mut hidden_weights);
    initialize_weights(&mut output_weights);

    // INITIALIZE BIAS
    let mut rng = rand::thread_rng();
    for i in 0..NUM_HIDDEN {
        hidden_layer_bias[i] = rng.gen_range(0.0..1.0);
    }
    for i in 0..NUM_OUTPUTS {
        output_layer_bias[i] = rng.gen_range(0.0..1.0);
    }

    let training_set_order = vec![0, 1, 2, 3];

    // TRAIN THE NETWORK
    for _epoch in 0..EPOCHS {
        for &i in &training_set_order {
            // FORWARD PASS

            // COMPUTE HIDDEN LAYER ACTIVATION
            for j in 0..NUM_HIDDEN {
                let mut activation = hidden_layer_bias[j];
                for k in 0..NUM_INPUTS {
                    activation += training_data[i][k] * hidden_weights[j][k];
                }
                hidden_layer[j] = sigmoid(activation);
            }

            // COMPUTE OUTPUT LAYER ACTIVATION
            for j in 0..NUM_OUTPUTS {
                let mut activation = output_layer_bias[j];
                for k in 0..NUM_HIDDEN {
                    activation += hidden_layer[k] * output_weights[j][k];
                }
                output_layer[j] = sigmoid(activation);
            }
            println!(
                "Input: {:?} -> Predicted output: {:?} (Expected: {:?})",
                training_data[i], output_layer, training_output[i]
            );

            // BACKPROPAGATION

            // COMPUTE CHANGE IN OUTPUT WEIGHTS
            let mut delta_output = vec![0.0; NUM_OUTPUTS];
            for b in 0..NUM_OUTPUTS {
                let error = training_output[i][b] - output_layer[b];
                delta_output[b] = error * sigmoid_derivative(output_layer[b]);
            }

            let mut delta_hidden = vec![0.0; NUM_HIDDEN];
            for m in 0..NUM_HIDDEN {
                let mut new_error = 0.0;
                for v in 0..NUM_OUTPUTS {
                    new_error += delta_output[v] * output_weights[v][m];
                }
                delta_hidden[m] = new_error * sigmoid_derivative(hidden_layer[m]);
            }

            // APPLY CHANGE IN OUTPUT WEIGHTS
            for c in 0..NUM_OUTPUTS {
                output_layer_bias[c] += delta_output[c] * learning_rate;
                for z in 0..NUM_HIDDEN {
                    output_weights[c][z] += hidden_layer[z] * delta_output[c] * learning_rate;
                }
            }

            // APPLY CHANGE IN HIDDEN WEIGHTS
            for p in 0..NUM_HIDDEN {
                hidden_layer_bias[p] += delta_hidden[p] * learning_rate;
                for o in 0..NUM_INPUTS {
                    hidden_weights[p][o] += training_data[i][o] * delta_hidden[p] * learning_rate;
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
