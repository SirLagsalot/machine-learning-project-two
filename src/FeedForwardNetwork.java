import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private int numLayers;
    private double learningRate;
    private ArrayList<Layer> layers;
    private IActivationFunction activationFunction;

    public FeedForwardNetwork(int numInputs, int numOutputs, int[] layerDimensions, double learningRate, IActivationFunction activationFunction) {
        super(numInputs, numOutputs);
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        this.initializeLayers(layerDimensions);
    }

    @Override
    public void train(List<Sample> samples) {
        int updateCount = 0;
        for (Sample sample : samples) {
            updateCount++;
            double networkOutput[] = this.forwardPropagate(sample.inputs);
            this.backPropagateError(sample.outputs);
            if (updateCount % 5 == 0) {
                this.updateWeights();
                updateCount = 0;
            }

            double error = 0.0;
            for (int i = 0; i < sample.outputs.length; i++) {
                error += Math.abs(networkOutput[i] - sample.outputs[i]);
            }

            System.out.println("Error: " + error + " ");
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        return this.forwardPropagate(inputs);
    }

    /**
     * Performs a single forward propagation through the network
     *
     * @param inputs: The input values to the function
     * @return values of the output layer neurons
     */
    private double[] forwardPropagate(double[] inputs) {

        // Set value of input layer to numInputs values
        for (int i = 0; i < this.numInputs; i++) {
            this.layers.get(0).getNeuron(i).setOutput(inputs[i]);
        }

        for (int i = 1; i < this.numLayers; i++) {              // Loop over each layer in the network
            Layer currentLayer = this.layers.get(i);
            Layer prevLayer = this.layers.get(i - 1);
            for (int j = 0; j < currentLayer.size; j++) {       // Loop over each neuron in the layer
                double neuronOutput = 0.0;
                Neuron currentNeuron = currentLayer.getNeuron(j);
                for (int k = 0; k < prevLayer.size; k++) {      // Loop over each of the current neuron's inputs
                    neuronOutput += currentNeuron.getWeight(k) * prevLayer.getNeuron(k).getOutput();
                }
                if (i != this.numLayers - 1) {                  // Apply activation function if not the output layer
                    neuronOutput = this.activationFunction.compute(neuronOutput + currentNeuron.getBias());
                }
                currentNeuron.setOutput(neuronOutput);
            }
        }

        // Collect values from output layer neurons
        double[] outputs = new double[this.numOutputs];
        Layer outputLayer = this.layers.get(this.numLayers - 1);
        for (int i = 0; i < this.numOutputs; i++) {
            outputs[i] = outputLayer.getNeuron(i).getOutput();
        }

        return outputs;
    }

    private void backPropagateError(double[] expectedOutput) {
        for (int i = this.numLayers - 1; i >= 0; i--) {
            Layer currentLayer = this.layers.get(i);
            List<Double> errors = new ArrayList<>();
            if (i != this.numLayers - 1) {
                for (int j = 0; j < currentLayer.size; j++) {
                    double error = 0.0;
                    List<Neuron> prevLayer = this.layers.get(i + 1).getNeurons();
                    for (Neuron neuron : prevLayer) {
                        error += neuron.getWeight(j) * neuron.getDelta();
                    }
                    errors.add(error);
                }
            } else {
                for (int j = 0; j < currentLayer.size; j++) {
                    Neuron neuron = currentLayer.getNeuron(j);
                    errors.add(expectedOutput[j] - neuron.getOutput());
                }
            }

            for (int j = 0; j < currentLayer.size; j++) {
                Neuron neuron = currentLayer.getNeuron(j);
                neuron.setDelta(errors.get(j) * this.activationFunction.computeDerivative(neuron.getOutput()));
            }
        }
    }

    private void updateWeights() {
        for (int i = 0; i < this.numLayers - 1; i++) {
            List<Double> inputs = new ArrayList<>();
            if (i != 0) {
                Layer prevLayer = this.layers.get(i - 1);
                for (int j = 0; j < prevLayer.size; j++) {
                    inputs.add(prevLayer.getNeuron(j).getOutput());
                }
                Layer currentLayer = this.layers.get(i);
                for (int j = 0; j < currentLayer.size; j++) {
                    Neuron neuron = currentLayer.getNeuron(j);
                    for (int k = 0; k < inputs.size(); k++) {
                        double updatedWeight = this.learningRate * neuron.getDelta() * inputs.get(k);
                        neuron.setWeight(k, updatedWeight);
                    }
                    neuron.setBias(neuron.getBias() + this.learningRate * neuron.getDelta());
                }
            } else {
                Layer inputLayer = this.layers.get(0);
                for (int j = 0; j < inputLayer.size; j++) {
                    inputs.add(inputLayer.getNeuron(j).getOutput());
                }
            }
        }
    }

    private void initializeLayers(int[] dimensions) {
        this.layers = new ArrayList<>(dimensions.length);
        this.numLayers = dimensions.length;

        this.layers.add(new Layer(dimensions[0], 0));                       // Input layer
        for (int i = 1; i < dimensions.length; i++) {
            this.layers.add(new Layer(dimensions[i], dimensions[i - 1]));   // Hidden and output layers
        }
    }
}

