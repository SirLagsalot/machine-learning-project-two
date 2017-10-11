import java.util.ArrayList;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private int numLayers;
    private double learningRate;
    private ArrayList<Layer> layers;
    private IActivationFunction activationFunction;

    public FeedForwardNetwork(int inputs, int outputs, int[] layerDimensions, double learningRate, IActivationFunction activationFunction) {
        super(inputs, outputs);
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        this.initializeLayers(layerDimensions);
    }

    @Override
    public void train(List<Sample> samples) {
        for (Sample sample : samples) {
            double networkOutput[] = this.execute(sample.inputs);
            this.backPropagate(sample.outputs, networkOutput);

            double error = 0.0;
            for (int i = 0; i < sample.outputs.length; i++) {
                error += Math.abs(networkOutput[i] - sample.outputs[i]);
            }

            double avgError = error / sample.outputs.length;
            System.out.println("Error: " + avgError);
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        return this.execute(inputs);
    }

    /**
     * Performs a single forward propagation through the network
     *
     * @param inputs: The input values to the function
     * @return values of the output layer neurons
     */
    private double[] execute(double[] inputs) {

        // Set value of input layer to inputs values
        for (int i = 0; i < this.inputs; i++) {
            this.layers.get(0).neurons[i].output = inputs[i];
        }

        // Loop through neurons in each layer computing output
        for (int k = 1; k < this.layers.size(); k++) {                      // Iterate over layers starting at first hidden
            for (int i = 0; i < this.layers.get(k).size; i++) {             // Iterate over neurons in each layer
                double neuronOutput = 0.0;
                for (int j = 0; j < this.layers.get(k - 1).size; j++) {
                    double weightJ = this.layers.get(k).neurons[i].weights[j];
                    double inputJ = this.layers.get(k - 1).neurons[j].output;
                    neuronOutput += weightJ * inputJ;
                }

                neuronOutput += this.layers.get(k).neurons[i].bias;         // Add bias

                this.layers.get(k).neurons[i].output = this.activationFunction.compute(neuronOutput);
            }
        }

        // Collect the output from the output layer
        Neuron[] outputLayer = this.getOutputLayer();

        double output[] = new double[outputs];
        for (int i = 0; i < this.outputs; i++) {
            output[i] = outputLayer[i].output;
        }

        return output;
    }

    private void backPropagate(double[] outputs, double[] networkOutput) {

        // Compute error deltas for output layer
        Neuron[] outputLayer = this.getOutputLayer();
        for (int i = 0; i < this.outputs; i++) {
            double error = outputs[i] - networkOutput[i];
            outputLayer[i].delta = error * this.activationFunction.computeDerivative(networkOutput[i]);
        }


        // Compute error deltas for hidden layers
        for (int k = this.numLayers - 2; k >= 0; k--) {
            for (int i = 0; i < this.layers.get(k).size; i++) {
                double error = 0.0;
                for (int j = 0; j < this.layers.get(k + 1).size; j++) {
                    error += this.layers.get(k + 1).neurons[j].delta * this.layers.get(k + 1).neurons[j].weights[i];
                }

                this.layers.get(k).neurons[i].delta = error * this.activationFunction.computeDerivative(this.layers.get(k).neurons[i].output);
            }

            this.updateWeights(k);
        }
    }

    private void updateWeights(int index) {
        for (int i = 0; i < this.layers.get(index + 1).size; i++) {
            for (int j = 0; j < this.layers.get(index).size; j++) {
                double delta = this.layers.get(index + 1).neurons[i].delta;
                double output = this.layers.get(index).neurons[j].output;
                this.layers.get(index + 1).neurons[i].weights[j] += this.learningRate * delta * output;
            }
            this.layers.get(index + 1).neurons[i].bias += this.learningRate * this.layers.get(index + 1).neurons[i].delta;
        }
    }

    private Neuron[] getOutputLayer() {
        return this.layers.get(this.numLayers - 1).neurons;
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

