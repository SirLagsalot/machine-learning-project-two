import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private ArrayList<Layer> layers;
    private boolean momentum;

    public FeedForwardNetwork(int inputs, int outputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                              double learningRate, boolean momentum, IActivationFunction activationFunc) {
        super(inputs, outputs);
        this.momentum = momentum;
        this.activationFunction = activationFunc;
        this.initializeNeurons(numHiddenLayers, numNeuronsPerHiddenLayer, learningRate, activationFunc);
    }

    @Override
    public void train(List<Sample> samples) {
        for (Sample sample : samples) {
            Double[] errors = this.forwardPropagation(sample);
            this.backPropagation(errors, sample.outputs);
        }
    }

    @Override
    public Double[] approximate(Double[] inputs) {
        return this.execute(inputs);
    }

    // Execute forward propagation, return error
    private Double[] forwardPropagation(Sample sample) {
        Double[] outputs = this.execute(sample.inputs);
        Double[] errors = new Double[this.outputs];
        for (int i = 0; i < this.outputs; i++) {
            errors[i] = sample.outputs[i] - outputs[i];
        }

        return errors;
    }

    private Double[] execute(Double[] inputs) {
        ArrayList<Double> outputs = new ArrayList<>(Arrays.asList(inputs));
        for (int i = 1; i < layers.size(); i++) {
            outputs = layers.get(i).execute(outputs);
        }
        return outputs.toArray(new Double[outputs.size()]);
    }


    private void backPropagation(Double[] errors, Double[] outputs) {
        // Output Layer
        ArrayList<Neuron> outputLayer = this.layers.get(this.layers.size()).getNeurons();
        for (int i = 0; i < outputLayer.size(); i++) {
            Double delta = errors[i] * this.activationFunction.computeDerivative(outputs[i]);
            outputLayer.get(i).setDelta(delta);
        }

        // Hidden Layers
        for (int k = this.layers.size() - 2; k >= 0; k--) {
            for (int i = 0; i < this.layers.get(k).getSize(); i++) {
                double error = 0.0;
                for (int j = 0; j < this.layers.get(k + 1).getSize(); j++) {
                    Neuron neuron = this.getNeron(k + 1, j);
                    error += neuron.getDelta() * neuron.getWeight(i);
                }

                this.getNeron(k, i).setDelta(error * this.activationFunction.computeDerivative(this.getNeron(k, i).getOutput()));
            }
        }
    }

    private Neuron getNeron(int layer, int index) {
        return this.layers.get(layer).getNeurons().get(index);
    }

    private void initializeNeurons(int numHidden, int numNodes, double learningRate, IActivationFunction activationFunc) {
        this.layers = new ArrayList<>(numHidden + 2);

        // Input layer
        this.layers.add(new Layer(this.inputs, 1, learningRate, activationFunc));

        // Hidden layers
        for (int i = 0; i < numHidden; i++) {
            this.layers.add(new Layer(numNodes, this.layers.get(i).getSize(), learningRate, activationFunc));
        }

        // Output layer
        this.layers.add(new Layer(this.outputs, this.layers.get(this.layers.size() - 1).getSize(), learningRate, activationFunc));
    }
}
