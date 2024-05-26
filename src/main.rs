use rand::Rng;

const NUM_INPUTS: f32 = 2.0;
const NUM_HIDDEN: f32 = 2.0;
const NUM_OUTPUTS: f32 = 1.0;
const NUM_TRAINING_SETS: f32 = 4.0;

const EPOCHS: i32 = 10000;

fn initialize_weights(weights: &mut Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    for row in weights.iter_mut() {
        for weight in row.iter_mut() {
            *weight = rng.gen_range(-1.0..1.0);
            // println!("{}", *weight);
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    x * (1.0 - x)
}

fn main() {
    let learning_rate = 0.1;

    let mut hidden_layer = vec![NUM_HIDDEN];
    let mut output_layer = vec![NUM_OUTPUTS];

    let mut hidden_layer_bias = vec![NUM_HIDDEN];
    let mut output_layer_bias = vec![NUM_OUTPUTS];

    let mut hidden_weights: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_INPUTS as usize]; NUM_HIDDEN as usize];
    let mut output_weights: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_HIDDEN as usize]; NUM_OUTPUTS as usize];

    let training_data: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_TRAINING_SETS as usize]; NUM_INPUTS as usize];

    let training_output: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_TRAINING_SETS as usize]; NUM_OUTPUTS as usize];

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
    // println!("Hidden layer weights: {:?}", hidden_weights);
    // println!("Output layer weights: {:?}", output_weights);

    // INITIALIZE BIAS
    let mut rng = rand::thread_rng();
    for i in 0..NUM_HIDDEN as usize {
        hidden_layer_bias[i] = rng.gen_range(-1.0..1.0);
    }
    for i in 0..NUM_OUTPUTS as usize {
        output_layer_bias[i] = rng.gen_range(-1.0..1.0);
    }
    // println!("Output layer bias: {:?}", output_layer_bias);

    let training_set_order = vec![0, 1, 2, 3];

    // TRAIN THE NETWORK
    for _epoch in 0..EPOCHS as usize {
        for x in 0..NUM_TRAINING_SETS as usize {
            let i = training_set_order[x];

            // FORWARD PASS

            // COMPUTE HIDDEN LAYER ACTIVATION
            for j in 0..NUM_HIDDEN as usize {
                let mut activation = hidden_layer_bias[j];
                for k in 0..NUM_INPUTS as usize {
                    activation += training_data[i][k] * hidden_weights[k][j];
                }
                hidden_layer[j] = sigmoid(activation);
            }

            // COMPUTE OUTPUT LAYER ACTIVATION
            for j in 0..NUM_OUTPUTS as usize {
                let mut activation = output_layer_bias[j];
                for k in 0..NUM_HIDDEN as usize {
                    activation += hidden_layer[k] * output_weights[k][j];
                }
                output_layer[j] = sigmoid(activation);
            }
            println!(
                "Input: {}\nOuput: {}\nPredicted output: {}\n",
                training_data[i][0], output_layer[0], training_output[i][0]
            );

            // BACKPROPAGATION

            // COMPUTE CHANGE IN OUTPUT WEIGHTS

            let mut delta_output = vec![NUM_OUTPUTS];

            for b in 0..NUM_OUTPUTS as usize {
                let error = training_output[i][b] - output_layer[b];
                delta_output[b] = error * sigmoid_derivative(output_layer[b]);
            }

            let mut delta_hidden = vec![NUM_HIDDEN];

            for m in 0..NUM_HIDDEN as usize {
                let mut new_error = 0.0;
                for v in 0..NUM_OUTPUTS as usize {
                    new_error += delta_output[v] * output_weights[m][v];
                }
                delta_hidden[m] = new_error * sigmoid_derivative(hidden_layer[m]);
            }

            // APPLY CHANGE IN OUPUT WEIGHTS
            for c in 0..NUM_OUTPUTS as usize {
                output_layer_bias[c] += delta_output[c] * learning_rate;
                for z in 0..NUM_HIDDEN as usize {
                    output_weights[z][c] += hidden_layer[z] * delta_output[c] * learning_rate;
                }
            }

            // APPLY CHANGE IN HIDDEN WEIGHTS
            for p in 0..NUM_HIDDEN as usize {
                hidden_layer_bias[p] += delta_hidden[p] * learning_rate;
                for o in 0..NUM_INPUTS as usize {
                    hidden_weights[o][p] += training_data[i][o] * delta_hidden[p] * learning_rate;
                }
            }
        }
    }
}
