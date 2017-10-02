import java.util.ArrayList;

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
    public void train() {

    }

    @Override
    public double approximate() {
        return 0;
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
        this.layers.add(new Layer(1, this.layers.get(this.layers.size()).getNumNodes(), activationFunc));
    }
}
