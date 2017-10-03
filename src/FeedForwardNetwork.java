import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForwardNetwork extends NeuralNetwork {

    private ArrayList<Layer> layers;
    private boolean momentum;

    public FeedForwardNetwork(int inputs, int outputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                              boolean momentum, IActivationFunction activationFunc) {
        super(inputs, outputs);
        this.momentum = momentum;
        this.initializeNeurons(numHiddenLayers, numNeuronsPerHiddenLayer, activationFunc);
    }

    @Override
    public void train(List<Sample> samples) {
        for (Sample sample : samples) {
            double error = this.forwardPropagation(sample);
            this.backPropagation(error);
        }
    }

    @Override
    public double approximate(Double[] inputs) {
        return this.execute(inputs);
    }

    // Execute forward propagation, return error
    private double forwardPropagation(Sample sample) {
        return this.execute(sample.inputs) - sample.output;
    }

    private double execute(Double[] inputs) {
        ArrayList<Double> outputs = new ArrayList<>(Arrays.asList(inputs));
        for (int i = 1; i < layers.size(); i++) {
            outputs = layers.get(i).execute(outputs);
        }
        return outputs.get(0);
    }


    private void backPropagation(double error) {

    }

    private void initializeNeurons(int numHidden, int numNodes, IActivationFunction activationFunc) {
        this.layers = new ArrayList<>(numHidden + 2);

        // Input layer
        this.layers.add(new Layer(this.inputs, 1, activationFunc));

        // Hidden layers
        for (int i = 0; i < numHidden; i++) {
            this.layers.add(new Layer(numNodes, this.layers.get(i).getNumNodes(), activationFunc));
        }

        // Output layer
        this.layers.add(new Layer(this.outputs, this.layers.get(this.layers.size() - 1).getNumNodes(), activationFunc));
    }
}
