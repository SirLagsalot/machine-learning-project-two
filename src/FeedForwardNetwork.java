import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private ArrayList<Layer> layers;
    private boolean momentum;
    private double learningRate;

    public FeedForwardNetwork(int inputs, int outputs, int numLayers, int numNeurons,
                              double learningRate, boolean momentum, IActivationFunction activationFunction) {
        super(inputs, outputs);
        this.momentum = momentum;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;
        this.initializeNeurons(numLayers, numNeurons);
    }

    @Override
    public void train(List<Sample> samples) {
        for (Sample sample : samples) {
            double[] outputs = this.forwardPropagation(sample.inputs);
            double error = (sample.outputs[0] - outputs[0]) * this.activationFunction.computeDerivative(outputs[0]);
            //   System.out.println(error);
            this.backPropagate(sample.outputs);
            this.updateWeights();
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        return this.forwardPropagation(inputs);
    }

    // Execute forward propagation, return error
    private double[] forwardPropagation(double[] inputs) {
        double[] networkOutputs = inputs;

        for (Layer layer : this.layers) {
            networkOutputs = layer.execute(networkOutputs);
        }

        return networkOutputs;
    }

    private void backPropagate(double[] expected) {
        for (int i = this.layers.size() - 1; i >= 0; i--) {
            Layer layer = this.layers.get(i);
            List<Double> errors = new ArrayList<>();

            if (i == this.layers.size() - 1) {                                  // Output layer
                for (int j = 0; j < layer.size(); j++) {
                    errors.add(expected[j] - layer.getOutput(j));
                }
            } else {                                            // Hidden layers
                for (int j = 0; j < layer.size(); j++) {
                    double error = 0.0;
                    Layer prevLayer = this.layers.get(i + 1);
                    for (int k = 0; k < prevLayer.size(); k++) {
                        error += prevLayer.getWeight(k, j) * prevLayer.getDelta(k);
                    }
                    errors.add(error);
                }
            }

            for (int j = 0; j < layer.size(); j++) {
                double delta = errors.get(j) * this.activationFunction.computeDerivative(layer.getOutput(j));
                layer.setDelta(j, delta);
            }
        }
    }

    private void updateWeights() {
        for (int i = 0; i < this.layers.size() - 1; i++) {
            System.out.println(Arrays.toString(this.layers.get(i).getDeltas()));
        }
    }

    private void initializeNeurons(int numLayers, int hiddenNodes) {
        this.layers = new ArrayList<>(numLayers);

        // First Layer
        this.layers.add(new Layer(hiddenNodes, inputs, activationFunction));

        // Hidden layers
        for (int i = 0; i < numLayers - 1; i++) {
            this.layers.add(new Layer(hiddenNodes, hiddenNodes, activationFunction));
        }

        // Output layer
        this.layers.add(new Layer(outputs, hiddenNodes, activationFunction));
    }
}
