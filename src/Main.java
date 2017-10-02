public class Main {

    public static void main(String[] args) {

        int inputs = 2;
        int outputs = 1;
        int hiddenLayers = 1;
        int nodesPerLayer = 3;
        boolean momentum = false;
        IActivationFunction activationFunction = new HyperbolicFunction();

        IFunctionApproximator MLP = new FeedForwardNetwork(inputs, outputs, hiddenLayers, nodesPerLayer, momentum, activationFunction);
    }
}
