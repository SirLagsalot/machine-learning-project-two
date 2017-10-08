import java.util.List;

public class Main {

    public static void main(String[] args) {

        int inputs = 2;
        int outputs = 1;
        int numHiddenLayers = 1;
        int nodesPerLayer = 3;
        double learningRate = 0.1;
        boolean momentum = false;
        IActivationFunction activationFunction = new SigmoidFunction();

        IFunctionApproximator MLP = new FeedForwardNetwork(inputs, outputs, numHiddenLayers, nodesPerLayer, learningRate, momentum, activationFunction);
        List<Sample> samples = SampleGenerator.generateEasySamples(10, 2);  //SampleGenerator.generateSamples(10000, inputs, 5, 1);

        MLP.train(samples);
    }
}
