public class FeedForwardNetwork extends NeuralNetwork {

    private int hiddenLayers;
    private int hiddenNodesPerLayer;
    private boolean momentum;

    private IActivationFunction activationFunction;

    public FeedForwardNetwork(int inputs, int outputs, int hiddenLayers, int hiddenNodesPerLayer, boolean momentum, IActivationFunction activationFunction) {
        super(inputs, outputs);
        this.hiddenLayers = hiddenLayers;
        this.hiddenNodesPerLayer = hiddenNodesPerLayer;
        this.momentum = momentum;
        this.activationFunction = activationFunction;
    }

    @Override
    public void train() {

    }
}
