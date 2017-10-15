import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private List<Layer> network;

    private double learningRate;
    private double momentum;

    private int batchSize;
    private int epochs;

    public FeedForwardNetwork(int[] networkDimensions, double learningRate, int batchSize, double momentum, IActivationFunction activationFunction, int epochs) {
        super(networkDimensions[0], networkDimensions[networkDimensions.length - 1]);
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.momentum = momentum;
        this.epochs = epochs;
        this.activationFunction = activationFunction;

        this.initializeNetwork(networkDimensions);
    }

    public void printDeltas() {
        this.network.forEach(layer -> layer.getNeurons().forEach(neuron -> System.out.println(neuron.getDelta())));
    }

    @Override
    public void train(List<Sample> samples) {

        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            for (Sample sample : samples) {
                double[] networkOutputs = this.forwardPropagate(sample.inputs);
                this.backPropagate(sample.outputs);
                this.updateWeights(sample.inputs);
                this.resetWeightDeltas();
                epochError += this.calculateTotalError(sample.outputs, networkOutputs);
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        return this.forwardPropagate(inputs);
    }

    private double[] forwardPropagate(double[] inputs) {
        double[] layerOutputs = inputs;

        // Propagate through hidden layers, not using the activation function for the final layer
        for (int i = 0; i < network.size(); i++) {
            boolean shouldActivate = (i != network.size() - 1);
            layerOutputs = network.get(i).execute(layerOutputs, shouldActivate);
        }

        return layerOutputs;
    }

    private void backPropagate(double[] expectedOutputs) {
        for (int i = this.network.size() - 1; i >= 0; i--) {
            Layer currentLayer = this.network.get(i);
            List<Double> errors = new ArrayList<>();

            if (i != this.network.size() - 1) {
                for (int j = 0; j < currentLayer.size; j++) {
                    double error = 0.0;
                    Layer prevLayer = this.network.get(i + 1);
                    for (int k = 0; k < prevLayer.size; k++) {
                        error += prevLayer.getNeuron(k).getWeight(j) * prevLayer.getNeuron(k).getDelta();
                    }
                    errors.add(error);
                }
            } else {    // Maybe this is the gradient bit and there needs to be a derivative here?
                for (int j = 0; j < currentLayer.size; j++) {
                    Neuron neuron = currentLayer.getNeuron(j);
                    double error = expectedOutputs[j] - neuron.getOutput();
                    errors.add(error);
                }
            }

            for (int j = 0; j < currentLayer.size; j++) {
                Neuron neuron = currentLayer.getNeuron(j);
                double delta = errors.get(j) * this.activationFunction.computeDerivative(neuron.getOutput());
                neuron.setDelta(delta);
            }
        }
    }

    private void updateWeights(double[] networkInputs) {
        double[] inputs;
        for (int i = 0; i < this.network.size(); i++) {
            if (i == 0) {
                inputs = networkInputs;
            } else {
                Layer prevLayer = this.network.get(i - 1);
                inputs = new double[prevLayer.size];
                for (int j = 0; j < prevLayer.size; j++) {
                    inputs[j] = prevLayer.getNeuron(j).getOutput();
                }
            }
            Layer currentLayer = this.network.get(i);
            for (int j = 0; j < currentLayer.size; j++) {
                Neuron currentNeuron = currentLayer.getNeuron(j);
                for (int k = 0; k < inputs.length; k++) {
                    currentNeuron.updateWeight(k, this.learningRate * currentNeuron.getDelta() * inputs[k]);
                }
                currentNeuron.updateBias(this.learningRate * currentNeuron.getDelta());
            }
        }
    }

    private void resetWeightDeltas() {
        this.network.forEach(layer -> layer.getNeurons().forEach(neuron -> neuron.setDelta(0.0)));
    }

    private double calculateTotalError(double[] networkOutputs, double[] expectedOutputs) {
        assert networkOutputs.length == expectedOutputs.length;

        double errorSum = 0.0;
        // Calculate the sum over the squared error for each output value
        for (int i = 0; i < networkOutputs.length; i++) {
            double error = networkOutputs[i] - expectedOutputs[i];
            errorSum += Math.pow(error, 2);
        }

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
    }

    // Initialize each layer of the network, in the input layer is created implicitly in execution and does not have a layer
    private void initializeNetwork(int[] networkDimensions) {
        this.network = new ArrayList<>(networkDimensions.length);
        for (int i = 1; i < networkDimensions.length; i++) {
            this.network.add(new Layer(networkDimensions[i], networkDimensions[i - 1], this.activationFunction));
        }
    }
}