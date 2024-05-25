use rand::Rng;

const NUM_INPUTS: f32 = 2.0;
const NUM_HIDDEN: f32 = 2.0;
const NUM_OUTPUTS: f32 = 1.0;
const NUM_TRAINING_SETS: f32 = 4.0;

fn main() {
    let learning_rate = 0.1;

    let hidden_layer = vec![NUM_HIDDEN];
    let output_layer = vec![NUM_OUTPUTS];

    let hidden_layer_bias = vec![NUM_HIDDEN];
    let output_layer_bias = vec![NUM_OUTPUTS];

    let hidden_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_INPUTS as usize]; NUM_HIDDEN as usize];
    let output_weights: Vec<Vec<f32>> = vec![vec![0.0; NUM_HIDDEN as usize]; NUM_OUTPUTS as usize];

    let training_data: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_TRAINING_SETS as usize]; NUM_INPUTS as usize];

    let training_data: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let training_output: Vec<Vec<f32>> =
        vec![vec![0.0; NUM_TRAINING_SETS as usize]; NUM_OUTPUTS as usize];

    let training_output: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
}
