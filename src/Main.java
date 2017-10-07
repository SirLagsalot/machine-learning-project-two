import java.util.List;

public class Main {

    public static void main(String[] args) {

        int inputs = 2;
        int outputs = 1;
        int hiddenLayers = 2;
        int nodesPerLayer = 300;
        double learningRate = 0.1;
        boolean momentum = false;
        IActivationFunction activationFunction = new HyperbolicFunction();

        IFunctionApproximator MLP = new FeedForwardNetwork(inputs, outputs, hiddenLayers, nodesPerLayer, learningRate, momentum, activationFunction);
        List<Sample> samples = SampleGenerator.generateSamples(10000, inputs, 5, 1);

        MLP.train(samples);
    }
}
