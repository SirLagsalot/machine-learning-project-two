import java.util.ArrayList;
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
            System.out.println(error);
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
        for (int i = this.layers.size() - 1; i > 0; i--) {
            Layer layer = this.layers.get(i);
            int length = layer.size() - 1;
            List<Double> errors = new ArrayList<>();

            if (i == length) {                                  // Output layer
                for (int j = 0; j < length; j++) {
                    errors.add(expected[j] - layer.getOutput(j));
                }
            } else {                                            // Hidden layers
                for (int j = 0; j < length; j++) {
                    double error = 0.0;
                    Layer prevLayer = this.layers.get(i + 1);
                    for (int k = 0; k < prevLayer.size() - 1; k++) {
                        error += prevLayer.getWeight(k, j) * prevLayer.getDelta(j);
                    }
                    errors.add(error);
                }
            }

            for (int j = 0; j < length; j++) {
                double delta = errors.get(j) * this.activationFunction.computeDerivative(layer.getOutput(j));
                layer.setDelta(j, delta);
            }
        }
    }

    private void updateWeights() {
        for (int i = 0; i < this.layers.size() - 1; i++) {

        }
    }


    private void backPropagation(double[] errors, double[] outputs) {
        // Output Layer
//        ArrayList<Neuron> outputLayer = this.layers.get(this.layers.size() - 1).getNeurons();
//        for (int i = 0; i < outputLayer.size(); i++) {
//            Double delta = errors[i] * this.activationFunction.computeDerivative(outputs[i]);
//            outputLayer.get(i).setDelta(delta);
//        }

//        // Hidden Layers
//        for (int k = this.layers.size() - 2; k >= 0; k--) {
//            for (int i = 0; i < this.layers.get(k).size(); i++) {
//                double error = 0.0;
//                for (int j = 0; j < this.layers.get(k + 1).size(); j++) {
//                    Neuron neuron = this.getNeron(k + 1, j);
//                    error += neuron.getDelta() * neuron.getWeight(i);
//                }
//
//                Neuron neuron = this.getNeron(k, i);
//                neuron.setDelta(error * this.activationFunction.computeDerivative(neuron.getOutput()));
//            }
//
//            for (int i = 0; i < this.layers.get(k + 1).size(); i++) {
//                for (int j = 0; j < this.layers.get(k).size(); j++) {
//                    double updatedWeight = this.learningRate * this.getNeron(k + 1, i).getDelta() * this.getNeron(k, j).getOutput();
//                    this.getNeron(k + 1, i).setWeight(j, updatedWeight);
//                }
//            }
//        }
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
