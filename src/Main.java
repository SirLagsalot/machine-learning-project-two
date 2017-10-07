import java.util.List;

public class Main {

    public static void main(String[] args) {

        int inputs = 4;
        int outputs = 1;
        int hiddenLayers = 1;
        int nodesPerLayer = 3;
        double learningRate = 0.01;
        boolean momentum = false;
        IActivationFunction activationFunction = new HyperbolicFunction();

        IFunctionApproximator MLP = new FeedForwardNetwork(inputs, outputs, hiddenLayers, nodesPerLayer, learningRate, momentum, activationFunction);
        List<Sample> samples = SampleGenerator.generateSamples(100000, inputs, 5, 1);

        MLP.train(samples);
    }
}
